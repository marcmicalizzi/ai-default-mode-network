param(
    [Parameter(Mandatory=$true)][string]$Folder,
    [Parameter(Mandatory=$true)][int[]]$WorkerIds,
    [int]$MaxSeconds = 3600
)
$ErrorActionPreference = 'Stop'
$target = Join-Path (Resolve-Path -LiteralPath $Folder).Path 'wddm-sampled.json'
if (Test-Path -LiteralPath $target) { throw 'Use a fresh telemetry output.' }
$clock = [Diagnostics.Stopwatch]::StartNew()
$samples = 0
$invalidSamples = 0
$dedicatedPeak = 0.0
$sharedPeak = 0.0
$workingSetPeak = 0.0
$reportedWorkingSetPeak = 0.0
$commitPeak = 0.0
$phases = @{}
$firstPhase = $null
$outcome = 'time_limit'
try {
    while ($clock.Elapsed.TotalSeconds -lt $MaxSeconds) {
        if (Test-Path -LiteralPath (Join-Path $Folder 'process.json')) { $outcome = 'worker_finished'; break }
        $processes = @(Get-Process -Id $WorkerIds -ErrorAction SilentlyContinue)
        $workingSetPeak = [Math]::Max($workingSetPeak, [double](($processes | Measure-Object WorkingSet64 -Sum).Sum))
        $reportedWorkingSetPeak = [Math]::Max($reportedWorkingSetPeak, [double](($processes | Measure-Object PeakWorkingSet64 -Sum).Sum))
        $commitPeak = [Math]::Max($commitPeak, [double](($processes | Measure-Object PrivateMemorySize64 -Sum).Sum))
        # Wildcard queries can contain invalid counters for unrelated processes
        # that just exited. Inspect only this worker's returned counter statuses.
        $values = (Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage' -MaxSamples 1 -ErrorAction SilentlyContinue).CounterSamples
        $matching = @($values | Where-Object {
            $name = $_.InstanceName
            @($WorkerIds | Where-Object { $name -like "pid_$($_)_*" }).Count -gt 0
        })
        if (@($matching | Where-Object { $_.Status -ne 0 }).Count) { $invalidSamples++; continue }
        if ($matching.Count) {
            $dedicated = [double](($matching | Where-Object { $_.Path.EndsWith('\dedicated usage') } | Measure-Object CookedValue -Sum).Sum)
            $shared = [double](($matching | Where-Object { $_.Path.EndsWith('\shared usage') } | Measure-Object CookedValue -Sum).Sum)
            $dedicatedPeak = [Math]::Max($dedicatedPeak, $dedicated)
            $sharedPeak = [Math]::Max($sharedPeak, $shared)
            $samples++
            $progress = Get-Content -LiteralPath (Join-Path $Folder 'progress.json') -Raw | ConvertFrom-Json
            $phase = $progress.phase
            if ($null -eq $firstPhase) { $firstPhase = $phase }
            if (-not $phases.ContainsKey($phase)) { $phases[$phase] = @{ dedicated_peak_bytes=0.0; shared_peak_bytes=0.0; samples=0 } }
            $phases[$phase].dedicated_peak_bytes = [Math]::Max($phases[$phase].dedicated_peak_bytes, $dedicated)
            $phases[$phase].shared_peak_bytes = [Math]::Max($phases[$phase].shared_peak_bytes, $shared)
            $phases[$phase].samples++
        }
        if (Test-Path -LiteralPath (Join-Path $Folder 'process.json')) { $outcome = 'worker_finished'; break }
    }
} catch {
    $outcome = 'counter_error'
    throw
} finally {
    [ordered]@{
        source='Windows GPU Process Memory performance counters'; sampled=$true;
        transient_peaks_may_be_missed=$true; worker_ids=$WorkerIds; first_observed_phase=$firstPhase;
        samples=$samples; invalid_worker_samples=$invalidSamples; seconds=$clock.Elapsed.TotalSeconds; outcome=$outcome;
        dedicated_peak_bytes=$(if ($samples) { $dedicatedPeak } else { $null });
        shared_peak_bytes=$(if ($samples) { $sharedPeak } else { $null });
        sampled_process_working_set_peak_bytes=$workingSetPeak;
        reported_process_peak_working_set_bytes=$reportedWorkingSetPeak;
        sampled_process_commit_peak_bytes=$commitPeak; phases=$phases
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $target -Encoding utf8
}
