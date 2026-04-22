# PowerShell: 在运行推理/评估脚本前执行: . .\scripts\set_env.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$Provider = if ($env:LLM_PROVIDER) { $env:LLM_PROVIDER } else { "vllm" }
$VllmPort = if ($env:VLLM_PORT) { $env:VLLM_PORT } else { "8000" }
$OllamaPort = if ($env:OLLAMA_PORT) { $env:OLLAMA_PORT } else { "11434" }
$OllamaHost = if ($env:OLLAMA_HOST_BIND) { $env:OLLAMA_HOST_BIND } else { "127.0.0.1" }

if ($Provider -eq "ollama") {
    $Port = $OllamaPort
    $Served = if ($env:OLLAMA_MODEL_TAG) { $env:OLLAMA_MODEL_TAG } elseif ($env:LLM_MODEL) { $env:LLM_MODEL } else { "gemma-4-26b-a4b-it-q4" }
    $DefaultBaseUrl = "http://${OllamaHost}:$Port/v1"
    $DefaultApiKey = "ollama"
} else {
    $Port = $VllmPort
    $Served = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "Qwen3.5-9B" }
    $DefaultBaseUrl = "http://127.0.0.1:$Port/v1"
    $DefaultApiKey = "EMPTY"
}

$env:LLM_BASE_URL = if ($env:LLM_BASE_URL) { $env:LLM_BASE_URL } else { $DefaultBaseUrl }
$env:LLM_MODEL = $Served
$env:LLM_API_KEY = if ($env:LLM_API_KEY) { $env:LLM_API_KEY } else { $DefaultApiKey }

Write-Host "已设置 LLM 环境变量:"
Write-Host "  LLM_PROVIDER = $Provider"
Write-Host "  LLM_BASE_URL = $($env:LLM_BASE_URL)"
Write-Host "  LLM_MODEL    = $($env:LLM_MODEL)"
Write-Host "  LLM_API_KEY  = $($env:LLM_API_KEY)"
Write-Host "  项目根目录   = $ProjectRoot"
