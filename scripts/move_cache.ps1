$source = "$env:USERPROFILE\.lightkurve"
$dest = "D:\.lightkurve"

Write-Host "Checking directories..."
if (-not (Test-Path $source)) {
    Write-Host "Source folder $source does not exist. Nothing to move."
    exit
}

if (Test-Path $dest) {
    Write-Host "Destination folder $dest already exists. Please delete it manually if you want to overwrite it."
    exit
}

Write-Host "Moving '$source' to '$dest'..."
# Move the folder
Move-Item -Path $source -Destination "D:\"

# Verify move
if (Test-Path $dest) {
    Write-Host "Folder moved successfully."
    
    # Create Junction
    Write-Host "Creating Junction link..."
    New-Item -ItemType Junction -Path $source -Target $dest
    
    Write-Host "Success! Your lightkurve data is now on D: drive, but programs will still find it on C:."
} else {
    Write-Host "Error: Failed to move the folder."
}
