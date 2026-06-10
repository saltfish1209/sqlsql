"""
CrossEncoder 训练（支持环境变量超参、逐步 loss CSV、按 epoch 存 checkpoint）。

训练北极星指标: NDCG@6 —— 衡量模型把 gold 列往前排的综合排序能力，
不依赖单点 top_k 阈值。top_k 仅在推理/部署阶段作为业务参数使用。
"""
from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from training.cross_encoder_early_stop import early_stop_after_eval
from training.cross_encoder_eval import evaluate_cross_encoder
from training.dataset_io import read_jsonl
from training.eval_schedule import should_run_mid_eval
from training.prepare_data import print_column_filter_report
from training.train_cross_encoder import (
    DEFAULT_DATALOADER_WORKERS,
    FILTER_REPORT_FILE,
    TRAIN_FILE_JSON,
    TRAIN_FILE_JSONL,
    _collate_records,
    _ensure_padding_token,
    _forward_scores,
    _get_dataloader_workers_from_env,
    _get_use_amp_from_env,
    _load_train_records,
    _resolve_train_file,
    compute_mixed_loss,
)

BASE_MODEL = str(settings.reranker_base_model)

NDCG_EVAL_K = 6
EVAL_EVERY_STEPS_AUTO = -1  # 按当前 epoch 步数的一半做中期 val；-1 为默认


@dataclass
class CrossEncoderTrainConfig:
    pair_lambda: float = 0.7
    pair_margin: float = 0.15
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    save_path: str = ""
    checkpoint_dir: str = ""
    loss_log_path: str = ""
    log_every_steps: int = 20
    eval_split: str = "val"
    eval_top_k: int = 6
    epoch_val_csv: str = ""
    early_stop_patience: int = 0  # 0 表示使用 figure.crossencoder.paths.get_early_stop_patience()
    early_stop_enabled: bool = True
    eval_every_steps: int = EVAL_EVERY_STEPS_AUTO  # -1=每 epoch 半程；0=关闭；>0=每 N 步


def _publish_checkpoint(src_dir: str, dest_dir: str) -> None:
    """将 best epoch 目录复制到最终 save_path。"""
    src = Path(src_dir)
    dest = Path(dest_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"checkpoint 不存在: {src_dir}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _prune_previous_best_checkpoint(previous_ckpt_path: str, current_ckpt_path: str) -> None:
    """只保留当前 best checkpoint，删除旧的 best 目录。"""
    if not previous_ckpt_path:
        return
    if os.path.abspath(previous_ckpt_path) == os.path.abspath(current_ckpt_path):
        return
    if os.path.isdir(previous_ckpt_path):
        shutil.rmtree(previous_ckpt_path, ignore_errors=True)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _default_eval_top_k() -> int:
    """NDCG 评估使用的 K 值（固定 6，不作为扫参变量）。"""
    return NDCG_EVAL_K


def _resolve_early_stop_patience(patience: int) -> int:
    """<=0 时从 paths 解析默认 patience。"""
    if patience > 0:
        return patience
    try:
        from figure.crossencoder.paths import get_early_stop_patience

        return get_early_stop_patience()
    except ImportError:
        return 2


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def config_from_env() -> CrossEncoderTrainConfig:
    save = os.getenv("NL2SQL_CE_SAVE_PATH", str(settings.cross_encoder_model)).strip()
    return CrossEncoderTrainConfig(
        pair_lambda=_env_float("NL2SQL_CE_PAIR_LAMBDA", 0.7),
        pair_margin=_env_float("NL2SQL_CE_PAIR_MARGIN", 0.15),
        epochs=_env_int("NL2SQL_CE_EPOCHS", 3),
        batch_size=_env_int("NL2SQL_CE_BATCH_SIZE", 16),
        lr=_env_float("NL2SQL_CE_LR", 2e-5),
        save_path=save,
        checkpoint_dir=os.getenv("NL2SQL_CE_CHECKPOINT_DIR", "").strip(),
        loss_log_path=os.getenv("NL2SQL_CE_LOSS_LOG", "").strip(),
        log_every_steps=_env_int("NL2SQL_CE_LOG_EVERY", 20),
        eval_split=os.getenv("NL2SQL_CE_EVAL_SPLIT", "val").strip() or "val",
        eval_top_k=_env_int("NL2SQL_CE_EVAL_TOP_K", _default_eval_top_k()),
        epoch_val_csv=os.getenv("NL2SQL_CE_EPOCH_VAL_CSV", "").strip(),
        early_stop_patience=_resolve_early_stop_patience(0),
        early_stop_enabled=os.getenv("NL2SQL_CE_EARLY_STOP", "1").strip().lower()
        not in {"0", "false", "no", "off"},
        eval_every_steps=_env_int("NL2SQL_CE_EVAL_EVERY_STEPS", EVAL_EVERY_STEPS_AUTO),
    )


def _append_csv(path: str, fieldnames: list[str], row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.is_file() or p.stat().st_size == 0
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_cross_encoder(config: CrossEncoderTrainConfig | None = None) -> str:
    cfg = config or config_from_env()
    cfg.early_stop_patience = _resolve_early_stop_patience(cfg.early_stop_patience)
    if not cfg.save_path:
        cfg.save_path = str(settings.cross_encoder_model)

    train_file = _resolve_train_file()
    if train_file is None:
        raise FileNotFoundError(
            f"训练数据不存在: {TRAIN_FILE_JSONL} 或 {TRAIN_FILE_JSON}，请先运行 prepare_data.py"
        )
    if not os.path.isdir(BASE_MODEL):
        raise FileNotFoundError(f"基础模型不存在: {BASE_MODEL}")

    if os.path.isfile(FILTER_REPORT_FILE):
        try:
            with open(FILTER_REPORT_FILE, "r", encoding="utf-8") as f:
                report = json.load(f)
            print_column_filter_report(report)
            print(f"  报告文件: {FILTER_REPORT_FILE}")
        except Exception as exc:
            print(f"[WARN] 列过滤报告读取失败: {type(exc).__name__}: {exc}")
    else:
        print(
            f"[WARN] 未找到列过滤报告 {FILTER_REPORT_FILE}，"
            "请先运行 prepare_data.py 以查看废弃列明细"
        )

    print(f"加载基座 Reranker: {BASE_MODEL}")
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=512, trust_remote_code=True)
    _ensure_padding_token(model)

    raw = _load_train_records(train_file)
    if not raw:
        raise RuntimeError(f"训练数据为空: {train_file}")
    print(f"训练样本条数: {len(raw)}")

    num_workers = _get_dataloader_workers_from_env()
    loader = DataLoader(
        raw,
        shuffle=True,
        batch_size=cfg.batch_size,
        collate_fn=_collate_records,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * 0.1)
    use_amp = _get_use_amp_from_env()

    print(f"训练 {cfg.epochs} epochs，总步数 {total_steps}")
    print(f"pair_lambda={cfg.pair_lambda} pair_margin={cfg.pair_margin}")
    print(f"保存路径: {cfg.save_path}")
    if cfg.checkpoint_dir:
        print(f"Checkpoint 目录: {cfg.checkpoint_dir}")
    if cfg.loss_log_path:
        print(f"Loss CSV: {cfg.loss_log_path}")
    eval_k = cfg.eval_top_k if cfg.eval_top_k > 0 else _default_eval_top_k()
    do_epoch_val = bool(cfg.epoch_val_csv)
    do_early_stop = cfg.early_stop_enabled and do_epoch_val and cfg.early_stop_patience > 0
    if cfg.eval_every_steps == 0:
        mid_eval_steps = 0
    elif cfg.eval_every_steps < 0:
        mid_eval_steps = max(1, steps_per_epoch // 2)
    else:
        mid_eval_steps = cfg.eval_every_steps
    if do_epoch_val:
        if mid_eval_steps > 0:
            print(
                f"val 评估: 每 epoch 约 {mid_eval_steps} 步(半程) + epoch 末尾  "
                f"NDCG@{eval_k} -> {cfg.epoch_val_csv}；第 1 epoch 中期 val 不计入早停"
            )
        else:
            print(f"val 评估: 每 epoch 末尾  NDCG@{eval_k} -> {cfg.epoch_val_csv}")
    if do_early_stop:
        print(
            f"早停: patience={cfg.early_stop_patience} "
            f"(连续 {cfg.early_stop_patience} 次 val NDCG@{eval_k} 未创新高则停止)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.train()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=max(total_steps, 1),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and device.type == "cuda")

    loss_fields = [
        "epoch",
        "step_in_epoch",
        "global_step",
        "loss",
        "point",
        "pair",
        "pair_lambda",
        "pair_margin",
    ]
    epoch_val_fields = [
        "epoch",
        "global_step_end",
        "pair_lambda",
        "pair_margin",
        "eval_k",
        "split",
        "ndcg",
        "mrr",
        "recall_at_k",
        "total",
        "success",
        "model_path",
        "is_best",
        "for_early_stop",
        "patience_counter",
        "stopped_early",
    ]

    global_step = 0
    best_ndcg = -1.0
    best_ckpt_path = ""
    best_epoch_label = ""
    patience_counter = 0
    stopped_early = False
    eval_count = 0

    def _run_eval(label: str, save_dir: str = "", *, for_early_stop: bool = True) -> bool:
        """直接用训练中的模型做 val；for_early_stop=False 时只记录指标，不参与早停。"""
        nonlocal best_ndcg, best_ckpt_path, best_epoch_label
        nonlocal patience_counter, stopped_early, eval_count
        eval_count += 1
        model.model.eval()
        try:
            with torch.no_grad():
                metrics = evaluate_cross_encoder(
                    split=cfg.eval_split, top_k=eval_k, model=model,
                )
        finally:
            model.model.train()

        current_ndcg = float(metrics["ndcg"])
        current_mrr = float(metrics["mrr"])
        current_recall = float(metrics["recall_at_k"])
        should_stop = False
        ckpt_path = ""

        if for_early_stop:
            is_best = current_ndcg > best_ndcg
            if is_best:
                best_ndcg = current_ndcg
                best_epoch_label = label
                patience_counter = 0
                if save_dir:
                    ckpt_path = os.path.join(save_dir, label)
                    os.makedirs(ckpt_path, exist_ok=True)
                    previous_best_ckpt = best_ckpt_path
                    model.save(ckpt_path)
                    _prune_previous_best_checkpoint(previous_best_ckpt, ckpt_path)
                    best_ckpt_path = ckpt_path
                    print(f"[{label}] 新最优，已保存 checkpoint: {ckpt_path}")
            elif do_early_stop:
                _, patience_counter, should_stop = early_stop_after_eval(
                    current_metric=current_ndcg,
                    best_metric=best_ndcg,
                    patience_counter=patience_counter,
                    patience=cfg.early_stop_patience,
                )
        else:
            is_best = False

        row = {
            "epoch": label,
            "global_step_end": global_step,
            "pair_lambda": cfg.pair_lambda,
            "pair_margin": cfg.pair_margin,
            "eval_k": eval_k,
            "split": cfg.eval_split,
            "ndcg": round(current_ndcg, 6),
            "mrr": round(current_mrr, 6),
            "recall_at_k": round(current_recall, 6),
            "total": metrics["total"],
            "success": metrics["success"],
            "model_path": ckpt_path or best_ckpt_path,
            "is_best": int(is_best),
            "for_early_stop": int(for_early_stop),
            "patience_counter": patience_counter,
            "stopped_early": int(should_stop),
        }
        _append_csv(cfg.epoch_val_csv, epoch_val_fields, row)
        es_note = "" if for_early_stop else " (不计入早停)"
        print(
            f"[{label}] val NDCG@{eval_k}: {current_ndcg:.4f}  "
            f"MRR: {current_mrr:.4f}  "
            f"Recall@{eval_k}: {current_recall:.2%} ({metrics['success']}/{metrics['total']})"
            f"{' [best]' if is_best else ''}"
            f"  patience={patience_counter}/{cfg.early_stop_patience}{es_note}"
        )

        if for_early_stop and do_early_stop and should_stop:
            stopped_early = True
            print(
                f"[EarlyStop] 连续 {patience_counter} 次 val "
                f"NDCG@{eval_k} 未超过 best={best_ndcg:.4f} ({best_epoch_label})，停止训练"
            )
        return should_stop

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(loader, start=1):
            global_step += 1
            point_pairs = [[it["question"], it["column"]] for it in batch]
            labels = torch.tensor(
                [float(it["label"]) for it in batch], dtype=torch.float32, device=device,
            )
            pair_records = [
                it for it in batch
                if int(it.get("label", 0)) == 0 and it.get("positive_column")
            ]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp and device.type == "cuda",
            ):
                point_scores = _forward_scores(model, point_pairs, device)
                if pair_records:
                    pos_pairs = [[it["question"], it["positive_column"]] for it in pair_records]
                    neg_pairs = [[it["question"], it["column"]] for it in pair_records]
                    pos_pair_scores = _forward_scores(model, pos_pairs, device)
                    neg_pair_scores = _forward_scores(model, neg_pairs, device)
                else:
                    pos_pair_scores = neg_pair_scores = None
                loss, point_loss, pair_loss = compute_mixed_loss(
                    point_scores,
                    labels,
                    pos_pair_scores=pos_pair_scores,
                    neg_pair_scores=neg_pair_scores,
                    margin=cfg.pair_margin,
                    pair_lambda=cfg.pair_lambda,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % cfg.log_every_steps == 0 or step == steps_per_epoch:
                step_loss = float(loss.detach().cpu())
                step_point = float(point_loss.cpu())
                step_pair = float(pair_loss.cpu())
                print(
                    f"[Epoch {epoch + 1}/{cfg.epochs} Step {step}/{steps_per_epoch}] "
                    f"loss={step_loss:.4f} point={step_point:.4f} pair={step_pair:.4f}"
                )
                if cfg.loss_log_path:
                    _append_csv(
                        cfg.loss_log_path,
                        loss_fields,
                        {
                            "epoch": epoch + 1,
                            "step_in_epoch": step,
                            "global_step": global_step,
                            "loss": round(step_loss, 6),
                            "point": round(step_point, 6),
                            "pair": round(step_pair, 6),
                            "pair_lambda": cfg.pair_lambda,
                            "pair_margin": cfg.pair_margin,
                        },
                    )

            if do_epoch_val and should_run_mid_eval(
                step=step,
                steps_per_epoch=steps_per_epoch,
                mid_eval_steps=mid_eval_steps,
            ):
                mid_label = f"E{epoch + 1}_S{step}"
                if _run_eval(
                    mid_label,
                    save_dir=cfg.checkpoint_dir,
                    for_early_stop=(epoch > 0),
                ):
                    break
        else:
            epoch_label = f"E{epoch + 1}"
            if do_epoch_val:
                if _run_eval(epoch_label, save_dir=cfg.checkpoint_dir):
                    break
            else:
                ckpt_path = cfg.save_path
                if cfg.checkpoint_dir:
                    ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch + 1}")
                    os.makedirs(ckpt_path, exist_ok=True)
                model.save(ckpt_path)
                print(f"[Epoch {epoch + 1}] 已保存: {ckpt_path}")
            continue
        break

    if best_ckpt_path:
        _publish_checkpoint(best_ckpt_path, cfg.save_path)
        print(
            f"最终模型已保存 (best={best_epoch_label}, NDCG@{eval_k}={best_ndcg:.4f}): "
            f"{cfg.save_path}"
        )
        if stopped_early:
            print(f"[EarlyStop] 已用 best checkpoint 覆盖 save_path（非最后一轮权重）")
    else:
        model.save(cfg.save_path)
        print(f"最终模型已保存: {cfg.save_path}")
    return cfg.save_path
