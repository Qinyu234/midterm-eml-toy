# Run a command with project-default Python 3.11.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$Py311 = "C:\Users\q234zhan\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $Py311)) {
    throw "Python 3.11 not found: $Py311"
}
& $Py311 @Args
