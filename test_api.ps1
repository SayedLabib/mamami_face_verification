# Test script for face recognition API
# PowerShell script to test the API endpoints

# Configuration
$API_BASE_URL = "http://localhost:8000"

# Colors for output
$GREEN = [ConsoleColor]::Green
$RED = [ConsoleColor]::Red
$CYAN = [ConsoleColor]::Cyan
$YELLOW = [ConsoleColor]::Yellow

Write-Host "Face Recognition API Test Script" -ForegroundColor $CYAN
Write-Host "=============================" -ForegroundColor $CYAN

# Function to test health endpoint
function Test-HealthEndpoint {
    Write-Host "`n[TESTING] Health Endpoint..." -ForegroundColor $CYAN
    try {
        $response = Invoke-RestMethod -Uri "$API_BASE_URL/health" -Method Get
        Write-Host "Status: " -NoNewline
        Write-Host "$($response.status)" -ForegroundColor $(if ($response.status -eq "healthy") { $GREEN } else { $YELLOW })
        Write-Host "Services:"
        foreach ($service in $response.services.PSObject.Properties) {
            Write-Host "  $($service.Name): " -NoNewline
            Write-Host "$($service.Value.status)" -ForegroundColor $(if ($service.Value.status -eq "healthy") { $GREEN } elseif ($service.Value.status -eq "warning") { $YELLOW } else { $RED })
        }
        return $true
    }
    catch {
        Write-Host "Error accessing health endpoint: $_" -ForegroundColor $RED
        return $false
    }
}

# Function to test vector-debug endpoint
function Test-VectorDebugEndpoint {
    Write-Host "`n[TESTING] Vector Debug Endpoint..." -ForegroundColor $CYAN
    try {
        $response = Invoke-RestMethod -Uri "$API_BASE_URL/vector-debug" -Method Get
        Write-Host "Collection Name: $($response.collection.name)" 
        Write-Host "Vector Count: $($response.vectors_count)" 
        
        if ($response.vectors_count -gt 0) {
            Write-Host "Sample Vectors: $($response.vectors_info.Count)" 
            foreach ($vec in $response.vectors_info) {
                Write-Host "  ID: $($vec.id), User ID: $($vec.user_id)" 
                Write-Host "  Vector Stats: Min: $($vec.vector_stats.min), Max: $($vec.vector_stats.max), Zeros: $($vec.vector_stats.zeros_percent)%" 
            }
        } else {
            Write-Host "No vectors stored yet" -ForegroundColor $YELLOW
        }
        return $true
    }
    catch {
        Write-Host "Error accessing vector-debug endpoint: $_" -ForegroundColor $RED
        return $false
    }
}

# Function to enroll a face
function Enroll-Face {
    param (
        [Parameter(Mandatory=$true)]
        [string]$ImagePath,
        [string]$UserId = $null,
        [string]$FullName = "Test User",
        [string]$Email = "test@example.com"
    )
    
    Write-Host "`n[ENROLLING] Face from $ImagePath..." -ForegroundColor $CYAN
    
    if (-not (Test-Path $ImagePath)) {
        Write-Host "Error: Image file does not exist" -ForegroundColor $RED
        return $null
    }
    
    try {
        $form = @{
            image = Get-Item $ImagePath
        }
        
        if ($UserId) {
            $form.user_id = $UserId
        }
        
        $form.full_name = $FullName
        $form.email = $Email
        
        $response = Invoke-RestMethod -Uri "$API_BASE_URL/extract-and-store" -Method Post -Form $form
        
        if ($response.success) {
            Write-Host "Success! User enrolled" -ForegroundColor $GREEN
            Write-Host "User ID: $($response.user_id)"
            Write-Host "Embedding ID: $($response.embedding_id)"
        } else {
            Write-Host "Failed to enroll user: $($response.message)" -ForegroundColor $RED
        }
        
        return $response
    }
    catch {
        Write-Host "Error enrolling face: $_" -ForegroundColor $RED
        return $null
    }
}

# Function to verify a face
function Verify-Face {
    param (
        [Parameter(Mandatory=$true)]
        [string]$ImagePath
    )
    
    Write-Host "`n[VERIFYING] Face from $ImagePath..." -ForegroundColor $CYAN
    
    if (-not (Test-Path $ImagePath)) {
        Write-Host "Error: Image file does not exist" -ForegroundColor $RED
        return $null
    }
    
    try {
        $form = @{
            image = Get-Item $ImagePath
        }
        
        $response = Invoke-RestMethod -Uri "$API_BASE_URL/verify-match" -Method Post -Form $form
        
        if ($response.success) {
            Write-Host "Verification result: " -NoNewline
            Write-Host $(if ($response.verified) { "MATCH FOUND" } else { "NO MATCH" }) -ForegroundColor $(if ($response.verified) { $GREEN } else { $YELLOW })
            
            if ($response.matches -and $response.matches.Count -gt 0) {
                Write-Host "Matches found: $($response.matches.Count)"
                foreach ($match in $response.matches) {
                    Write-Host "  User ID: $($match.user_id), Score: $($match.similarity_score)" -ForegroundColor $(
                        if ($match.similarity_score -gt 0.8) { $GREEN }
                        elseif ($match.similarity_score -gt 0.5) { $YELLOW }
                        else { $CYAN }
                    )
                }
            } else {
                Write-Host "No matches found" -ForegroundColor $YELLOW
            }
        } else {
            Write-Host "Verification failed: $($response.message)" -ForegroundColor $RED
        }
        
        return $response
    }
    catch {
        Write-Host "Error verifying face: $_" -ForegroundColor $RED
        return $null
    }
}

# Function to compare two faces directly
function Compare-Faces {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Image1Path,
        
        [Parameter(Mandatory=$true)]
        [string]$Image2Path
    )
    
    Write-Host "`n[COMPARING] Faces directly..." -ForegroundColor $CYAN
    Write-Host "Image 1: $Image1Path"
    Write-Host "Image 2: $Image2Path"
    
    if (-not (Test-Path $Image1Path) -or -not (Test-Path $Image2Path)) {
        Write-Host "Error: One or both image files do not exist" -ForegroundColor $RED
        return $null
    }
    
    try {
        $form = @{
            image1 = Get-Item $Image1Path
            image2 = Get-Item $Image2Path
        }
        
        $response = Invoke-RestMethod -Uri "$API_BASE_URL/compare-vectors" -Method Post -Form $form
        
        if ($response.success) {
            Write-Host "Comparison result: " -NoNewline
            Write-Host $(if ($response.is_match) { "MATCH" } else { "NO MATCH" }) -ForegroundColor $(if ($response.is_match) { $GREEN } else { $YELLOW })
            Write-Host "Similarity score: $($response.similarity)" -ForegroundColor $(
                if ($response.similarity -gt 0.8) { $GREEN }
                elseif ($response.similarity -gt 0.5) { $YELLOW }
                else { $CYAN }
            )
            Write-Host "Euclidean distance: $($response.euclidean_distance)"
            Write-Host "Threshold: $($response.threshold)"
        } else {
            Write-Host "Comparison failed: $($response.message)" -ForegroundColor $RED
        }
        
        return $response
    }
    catch {
        Write-Host "Error comparing faces: $_" -ForegroundColor $RED
        return $null
    }
}

# Main menu
function Show-Menu {
    Write-Host "`n==== MENU ====" -ForegroundColor $CYAN
    Write-Host "1. Check API health"
    Write-Host "2. Check vector database status"
    Write-Host "3. Enroll a face"
    Write-Host "4. Verify a face"
    Write-Host "5. Compare two faces directly"
    Write-Host "0. Exit"
    Write-Host "=============="
    
    $choice = Read-Host "Enter your choice"
    
    switch ($choice) {
        "1" { Test-HealthEndpoint }
        "2" { Test-VectorDebugEndpoint }
        "3" {
            $imagePath = Read-Host "Enter path to image file"
            $userId = Read-Host "Enter user ID (leave blank for auto-generated)"
            $fullName = Read-Host "Enter full name (leave blank for 'Test User')"
            $email = Read-Host "Enter email (leave blank for 'test@example.com')"
            
            if ([string]::IsNullOrWhiteSpace($userId)) { $userId = $null }
            if ([string]::IsNullOrWhiteSpace($fullName)) { $fullName = "Test User" }
            if ([string]::IsNullOrWhiteSpace($email)) { $email = "test@example.com" }
            
            Enroll-Face -ImagePath $imagePath -UserId $userId -FullName $fullName -Email $email
        }
        "4" {
            $imagePath = Read-Host "Enter path to image file"
            Verify-Face -ImagePath $imagePath
        }
        "5" {
            $image1Path = Read-Host "Enter path to first image file"
            $image2Path = Read-Host "Enter path to second image file"
            Compare-Faces -Image1Path $image1Path -Image2Path $image2Path
        }
        "0" { return $false }
        default { Write-Host "Invalid choice. Please try again." -ForegroundColor $RED }
    }
    
    return $true
}

# Run the main loop
$continue = $true
while ($continue) {
    $continue = Show-Menu
}

Write-Host "`nExiting test script." -ForegroundColor $CYAN
