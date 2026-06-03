param(
  [string]$ServerUrl = "http://127.0.0.1:18080",
  [string]$Message = "привет",
  [string]$SystemPrompt = "You are a helpful assistant. Reply only to the user. Do not describe the conversation.",
  [int]$MaxTokens = 128,
  [double]$Temperature = 0.2,
  [switch]$RawJson,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Escape-HarmonyText {
  param([string]$Text)
  if ($null -eq $Text) { return "" }
  return $Text.Replace("<|start|>", "").Replace("<|end|>", "").Replace("<|message|>", "").Replace("<|channel|>", "").Replace("<|return|>", "")
}

function New-GptOssPrompt {
  param(
    [string]$SystemPrompt,
    [string]$UserMessage
  )
  $cleanSystem = Escape-HarmonyText $SystemPrompt
  $cleanUser = Escape-HarmonyText $UserMessage
  return "<|start|>system<|message|>$cleanSystem<|end|><|start|>user<|message|>$cleanUser<|end|><|start|>assistant<|channel|>final<|message|>"
}

$prompt = New-GptOssPrompt -SystemPrompt $SystemPrompt -UserMessage $Message
$payload = @{
  prompt = $prompt
  n_predict = $MaxTokens
  temperature = $Temperature
  stop = @("<|return|>", "<|end|>", "<|start|>")
}

$payloadJson = $payload | ConvertTo-Json -Depth 8
if ($DryRun) {
  Write-Output $payloadJson
  return
}

$endpoint = $ServerUrl.TrimEnd("/") + "/completion"
$body = [System.Text.Encoding]::UTF8.GetBytes($payloadJson)
$response = Invoke-WebRequest `
  -UseBasicParsing `
  -Uri $endpoint `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $body `
  -TimeoutSec 600

if ($RawJson) {
  Write-Output $response.Content
  return
}

$parsed = $response.Content | ConvertFrom-Json
Write-Output ([string]$parsed.content).Trim()
