# PowerShell: 在运行推理/评估脚本前执行: . .\scripts\set_env.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$Port = if ($env:VLLM_PORT) { $env:VLLM_PORT } else { "8000" }
$Served = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "Qwen3.5-9B" }

$env:LLM_BASE_URL = if ($env:LLM_BASE_URL) { $env:LLM_BASE_URL } else { "http://127.0.0.1:$Port/v1" }
$env:LLM_MODEL = $Served
$env:LLM_API_KEY = if ($env:LLM_API_KEY) { $env:LLM_API_KEY } else { "EMPTY" }

Write-Host "已设置 LLM 环境变量:"
Write-Host "  LLM_BASE_URL = $($env:LLM_BASE_URL)"
Write-Host "  LLM_MODEL    = $($env:LLM_MODEL)"
Write-Host "  LLM_API_KEY  = $($env:LLM_API_KEY)"
Write-Host "  项目根目录   = $ProjectRoot"
