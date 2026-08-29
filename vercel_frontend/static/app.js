/**
 * Rice Grain Quality Analyzer — Frontend Application
 * ====================================================
 * Handles image upload, API communication, and results rendering.
 */

(function () {
    'use strict';

    // ============================================================
    // DOM Elements
    // ============================================================
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const btnRemove = document.getElementById('btn-remove');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnNew = document.getElementById('btn-new-analysis');
    const btnRetry = document.getElementById('btn-retry');

    const uploadSection = document.getElementById('upload-section');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    const errorDisplay = document.getElementById('error-display');

    // State
    let selectedFile = null;
    let overlayImageSrc = null;
    let originalImageSrc = null;
    let defectChart = null;
    let qualityGaugeChart = null;
    let priceChart = null;

    // ============================================================
    // API Configuration
    // ============================================================
    // Leave empty ('') if using Vercel rewrites, OR set to your Render URL:
    // e.g., const API_BASE = 'https://grainwise-backend.onrender.com';
    const API_BASE = '';

    // ============================================================
    // Initialize
    // ============================================================
    function init() {
        setupUploadHandlers();
        setupButtonHandlers();
        loadVarieties();
    }

    // ============================================================
    // Variety Dropdown
    // ============================================================
    function loadVarieties() {
        fetch(API_BASE + '/api/varieties')
            .then(r => r.json())
            .then(data => {
                const select = document.getElementById('variety-select');
                data.varieties.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.key;
                    opt.textContent = `${v.display_name} (${v.category})`;
                    select.appendChild(opt);
                });
            })
            .catch(() => {
                // Fallback: varieties will need manual input
            });
    }

    // ============================================================
    // File Upload Handlers
    // ============================================================
    function setupUploadHandlers() {
        // Click to browse
        uploadZone.addEventListener('click', () => fileInput.click());

        // File selection
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    function handleFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|tiff)$/i)) {
            alert('Please select a valid image file (JPG, PNG, BMP, or TIFF)');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            originalImageSrc = e.target.result;
            uploadZone.style.display = 'none';
            imagePreview.style.display = 'block';
            btnAnalyze.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // ============================================================
    // Button Handlers
    // ============================================================
    function setupButtonHandlers() {
        // Remove image
        btnRemove.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUpload();
        });

        // Analyze
        btnAnalyze.addEventListener('click', () => {
            if (selectedFile) analyzeImage();
        });

        // New analysis
        btnNew.addEventListener('click', () => {
            resetUpload();
            resultsSection.style.display = 'none';
            errorDisplay.style.display = 'none';
            uploadSection.style.display = 'block';
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        // Retry
        btnRetry.addEventListener('click', () => {
            errorDisplay.style.display = 'none';
            uploadSection.style.display = 'block';
        });

        // Image toggle buttons
        document.getElementById('btn-overlay').addEventListener('click', function () {
            setActiveToggle(this);
            document.getElementById('result-image').src = overlayImageSrc || '';
        });

        document.getElementById('btn-original').addEventListener('click', function () {
            setActiveToggle(this);
            document.getElementById('result-image').src = originalImageSrc || '';
        });
    }

    function setActiveToggle(activeBtn) {
        document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        activeBtn.classList.add('active');
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImg.src = '';
        uploadZone.style.display = 'block';
        imagePreview.style.display = 'none';
        btnAnalyze.disabled = true;
    }

    // ============================================================
    // Analysis
    // ============================================================
    function analyzeImage() {
        // Build form data
        const formData = new FormData();
        formData.append('image', selectedFile);

        const sampleWeight = document.getElementById('sample-weight').value;
        const arucoLengthEl = document.getElementById('aruco-length');
        const arucoWidthEl = document.getElementById('aruco-width');
        const arucoLength = arucoLengthEl ? arucoLengthEl.value : '10';
        const arucoWidth = arucoWidthEl ? arucoWidthEl.value : '10';
        const variety = document.getElementById('variety-select').value;
        const storageTemp = document.getElementById('storage-temp').value;
        const storageHumidity = document.getElementById('storage-humidity').value;

        if (sampleWeight) formData.append('sample_weight', sampleWeight);
        if (arucoLength) formData.append('aruco_length', arucoLength);
        if (arucoWidth) formData.append('aruco_width', arucoWidth);
        if (variety) formData.append('variety', variety);
        if (storageTemp) formData.append('storage_temp', storageTemp);
        if (storageHumidity) formData.append('storage_humidity', storageHumidity);

        // Show loading
        showLoading();

        // Start loading step animation
        animateLoadingSteps();

        // Send request
        fetch(API_BASE + '/api/analyze', {
            method: 'POST',
            body: formData,
        })
            .then(async r => {
                const contentType = r.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) {
                    if (!r.ok) {
                        const data = await r.json();
                        throw data;
                    }
                    return r.json();
                } else {
                    const text = await r.text();
                    throw new Error(`Server returned an invalid response (Status ${r.status}). Ensure the server is running and the image size is reasonable.`);
                }
            })
            .then(report => {
                hideLoading();
                if (report.status === 'error') {
                    showError(report.error || 'Analysis failed');
                } else {
                    displayResults(report);
                }
            })
            .catch(err => {
                hideLoading();
                showError(err.error || err.message || 'An unexpected error occurred');
            });
    }

    // ============================================================
    // Loading Animation
    // ============================================================
    function showLoading() {
        uploadSection.style.display = 'none';
        loadingOverlay.style.display = 'flex';
        resultsSection.style.display = 'none';
        errorDisplay.style.display = 'none';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }

    function animateLoadingSteps() {
        const steps = document.querySelectorAll('.loading-step');
        let current = 0;

        steps.forEach(s => { s.classList.remove('active', 'done'); });
        steps[0].classList.add('active');

        const interval = setInterval(() => {
            if (current < steps.length) {
                steps[current].classList.remove('active');
                steps[current].classList.add('done');
            }
            current++;
            if (current < steps.length) {
                steps[current].classList.add('active');
            } else {
                clearInterval(interval);
            }
        }, 1200);

        // Store interval for cleanup
        window._loadingInterval = interval;
    }

    function showError(message) {
        uploadSection.style.display = 'none';
        errorDisplay.style.display = 'block';
        document.getElementById('error-message').textContent = message;
    }

    // ============================================================
    // Display Results
    // ============================================================
    function displayResults(report) {
        resultsSection.style.display = 'block';
        uploadSection.style.display = 'none';

        // Set images
        overlayImageSrc = report.overlay_base64 || '';
        originalImageSrc = report.original_base64 || '';
        document.getElementById('result-image').src = overlayImageSrc;
        setActiveToggle(document.getElementById('btn-overlay'));

        // Summary cards
        renderVarietyCard(report);
        renderGradeCard(report);
        renderShelfCard(report);
        renderPriceCard(report);

        // Detail panels
        renderImageStats(report);
        renderDefectChart(report);
        renderPriceDetails(report);
        renderMoisturePanel(report);

        // Scroll to results
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // ---- Summary Cards ---- //

    function renderVarietyCard(report) {
        const v = report.variety || {};
        document.getElementById('variety-name').textContent = v.display_name || 'Unknown';
        document.getElementById('variety-category').textContent =
            `Category: ${(v.category || 'N/A').charAt(0).toUpperCase() + (v.category || '').slice(1)} grain`;
    }

    function renderGradeCard(report) {
        const q = report.quality || {};
        const score = q.quality_score || 0;
        const grade = q.faq_grade || 'Unknown';

        document.getElementById('faq-grade').textContent = grade;
        document.getElementById('quality-score-text').textContent = `Quality Score: ${score.toFixed(1)}/100`;

        const gradeEl = document.getElementById('grade-card');
        gradeEl.className = 'summary-card grade-card';
        if (grade === 'Grade A' || grade === 'Premium') {
            gradeEl.classList.add('grade-a');
        } else if (grade === 'Rejected') {
            gradeEl.classList.add('grade-reject');
        }
    }

    function renderShelfCard(report) {
        const sl = report.shelf_life || {};
        const months = sl.shelf_life_months;
        document.getElementById('shelf-life-value').textContent =
            months !== undefined ? `${months.toFixed(1)} mo` : '—';
        document.getElementById('shelf-life-conditions').textContent =
            `At ${sl.storage_conditions?.temperature || 25}°C, ${sl.storage_conditions?.humidity || 60}% RH`;
    }

    function renderPriceCard(report) {
        const p = report.pricing || {};
        document.getElementById('price-value').textContent =
            p.recommended_price ? `₹${p.recommended_price.toLocaleString('en-IN')}` : '—';
    }

    // ---- Image Stats ---- //

    function renderImageStats(report) {
        const s = report.segmentation || {};
        document.getElementById('grain-count').textContent = s.total_grains || '—';
        document.getElementById('avg-length').textContent =
            s.mean_length_mm ? `${s.mean_length_mm} mm` : '—';
        document.getElementById('avg-width').textContent =
            s.mean_width_mm ? `${s.mean_width_mm} mm` : '—';
        document.getElementById('avg-ar').textContent =
            s.mean_aspect_ratio || '—';
    }

    // ---- Defect Chart ---- //

    function renderDefectChart(report) {
        let defects = report.defects?.mutually_exclusive_percentages;
        if (!defects) {
            defects = report.defects?.multilabel_percentages || report.defects?.percentages || {};
        }
        
        const container = document.querySelector('.chart-container');

        // Destroy old chart
        if (defectChart) {
            defectChart.destroy();
            defectChart = null;
        }

        const labels = [];
        const data = [];
        const colors = {
            whole: '#34d399',
            broken: '#f87171',
            chalky: '#fbbf24',
            damaged: '#fb923c',
            discolored: '#c084fc',
            foreign: '#94a3b8',
        };
        const bgColors = [];

        for (const [key, val] of Object.entries(defects)) {
            if (val > 0) {
                const displayName = key === 'whole' ? 'Good Grain' : key.charAt(0).toUpperCase() + key.slice(1);
                labels.push(displayName);
                data.push(val);
                bgColors.push(colors[key] || '#6b7280');
            }
        }

        if (data.length === 0) {
            data.push(100);
            labels.push('Good Grain');
            bgColors.push(colors.whole);
        }

        const canvas = document.getElementById('defect-chart');
        defectChart = new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: bgColors,
                    borderColor: 'rgba(10, 11, 15, 0.8)',
                    borderWidth: 2,
                    hoverOffset: 8,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '60%',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(20, 22, 34, 0.95)',
                        titleColor: '#f0ece4',
                        bodyColor: '#a8a0b4',
                        borderColor: 'rgba(212, 165, 116, 0.2)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: (ctx) => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`,
                        },
                    },
                },
            },
        });

        // Build legend
        const legendContainer = document.getElementById('defect-legend');
        legendContainer.innerHTML = labels.map((label, i) =>
            `<div class="legend-item">
                <span class="legend-dot" style="background:${bgColors[i]}"></span>
                <span>${label}: ${data[i].toFixed(1)}%</span>
            </div>`).join('');
    }

    // ---- Price Details ---- //

    function renderPriceDetails(report) {
        const p = report.pricing || {};
        const breakdown = document.getElementById('price-breakdown');

        let html = `
            <div class="price-row">
                <span class="label">Base Price (${p.faq_grade || '—'})</span>
                <span class="value">₹${(p.base_price || 0).toLocaleString('en-IN')}</span>
            </div>`;

        // Deductions
        const deductions = p.deductions || {};
        const deductionItems = deductions.details || deductions.items || {};

        if (typeof deductionItems === 'object') {
            for (const [key, val] of Object.entries(deductionItems)) {
                if (typeof val === 'number' && val > 0) {
                    html += `
                        <div class="price-row">
                            <span class="label">${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} deduction</span>
                            <span class="deduction">−₹${val.toLocaleString('en-IN')}</span>
                        </div>`;
                }
            }
        }

        const totalDeduction = deductions.total || deductions.total_deduction || 0;
        if (totalDeduction > 0) {
            html += `
                <div class="price-row">
                    <span class="label">Total Deductions</span>
                    <span class="deduction">−₹${totalDeduction.toLocaleString('en-IN')}</span>
                </div>`;
        }

        html += `
            <div class="price-row">
                <span class="label">Recommended Price</span>
                <span class="value">₹${(p.recommended_price || 0).toLocaleString('en-IN')}/quintal</span>
            </div>`;

        breakdown.innerHTML = html;

        // Price comparison chart
        if (priceChart) {
            priceChart.destroy();
            priceChart = null;
        }

        const comparison = p.grade_comparison || p.comparison || {};
        if (Object.keys(comparison).length > 0) {
            const gradeLabels = Object.keys(comparison).map(k =>
                k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
            );
            const gradeValues = Object.values(comparison);
            const barColors = gradeLabels.map(l => {
                if (l.includes('A')) return '#34d399';
                if (l.includes('B')) return '#60a5fa';
                if (l.includes('Common')) return '#fbbf24';
                return '#f87171';
            });

            priceChart = new Chart(document.getElementById('price-chart'), {
                type: 'bar',
                data: {
                    labels: gradeLabels,
                    datasets: [{
                        data: gradeValues,
                        backgroundColor: barColors.map(c => c + '40'),
                        borderColor: barColors,
                        borderWidth: 2,
                        borderRadius: 6,
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(20, 22, 34, 0.95)',
                            titleColor: '#f0ece4',
                            bodyColor: '#a8a0b4',
                            callbacks: {
                                label: (ctx) => ` ₹${ctx.parsed.y.toLocaleString('en-IN')}/quintal`,
                            },
                        },
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { color: '#a8a0b4', font: { size: 11 } },
                        },
                        y: {
                            grid: { color: 'rgba(212, 165, 116, 0.06)' },
                            ticks: {
                                color: '#6b6580',
                                callback: v => '₹' + v.toLocaleString('en-IN'),
                            },
                        },
                    },
                },
            });
        }
    }

    // ---- Moisture Panel ---- //

    function renderMoisturePanel(report) {
        const q = report.quality || {};
        const moisture = (q.moisture_pct !== undefined && q.moisture_pct !== null) ? q.moisture_pct : null;
        const source = report.moisture_source || 'unknown';
        const noWeight = (source === 'default_estimate' || source === 'unknown' || moisture === null);

        // Value
        const moistureValueEl = document.getElementById('moisture-value');
        if (noWeight) {
            moistureValueEl.textContent = 'N/A';
            moistureValueEl.style.fontSize = '1.4rem';
        } else {
            moistureValueEl.textContent = `${moisture.toFixed(1)}%`;
            moistureValueEl.style.fontSize = '';
        }

        // Circle animation
        const arc = document.getElementById('moisture-arc');
        const circumference = 314; // 2 * pi * 50
        const displayMoisture = noWeight ? 0 : moisture;
        const pct = Math.min(displayMoisture / 20, 1); // Scale: 0-20% → 0-100%
        arc.style.strokeDashoffset = circumference * (1 - pct);

        // Color based on level
        if (noWeight) {
            arc.style.stroke = '#6b7280'; // Gray for unknown
        } else if (displayMoisture <= 12) {
            arc.style.stroke = '#34d399'; // Good
        } else if (displayMoisture <= 14) {
            arc.style.stroke = '#fbbf24'; // Acceptable
        } else {
            arc.style.stroke = '#f87171'; // High
        }

        // Source
        const sourceMap = {
            weight_based: 'Weight Measurement',
            default_estimate: 'Not Provided — Enter Weight',
            visual_estimate: 'Visual Estimate',
            unknown: 'Not Provided — Enter Weight'
        };
        document.getElementById('moisture-source').textContent = sourceMap[source] || source;

        // Status
        const statusEl = document.getElementById('moisture-status');
        if (noWeight) {
            statusEl.textContent = '— No Sample Weight';
            statusEl.style.color = '#9ca3af';
        } else if (displayMoisture <= 12) {
            statusEl.textContent = '✓ Optimal';
            statusEl.style.color = '#34d399';
        } else if (displayMoisture <= 14) {
            statusEl.textContent = '⚠ Acceptable';
            statusEl.style.color = '#fbbf24';
        } else {
            statusEl.textContent = '✗ Too High';
            statusEl.style.color = '#f87171';
        }
    }

    // ============================================================
    // PWA Install Banner
    // ============================================================
    let deferredPrompt = null;
    const installBanner = document.getElementById('install-banner');
    const btnInstallApp = document.getElementById('btn-install-app');
    const btnDismiss    = document.getElementById('btn-dismiss-install');
    const iosHint       = document.getElementById('ios-hint');
    const btnDismissIos = document.getElementById('btn-dismiss-ios');

    // Android Chrome — capture the native install prompt before it shows
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        if (installBanner) installBanner.style.display = 'flex';
    });

    // User taps Install
    if (btnInstallApp) {
        btnInstallApp.addEventListener('click', async () => {
            if (!deferredPrompt) return;
            deferredPrompt.prompt();
            const { outcome } = await deferredPrompt.userChoice;
            deferredPrompt = null;
            if (installBanner) installBanner.style.display = 'none';
        });
    }

    // User dismisses banner
    if (btnDismiss) {
        btnDismiss.addEventListener('click', () => {
            if (installBanner) installBanner.style.display = 'none';
        });
    }

    // After successful install, hide banner
    window.addEventListener('appinstalled', () => {
        if (installBanner) installBanner.style.display = 'none';
        deferredPrompt = null;
    });

    // iOS Safari — no beforeinstallprompt, show manual hint instead
    function isIosSafari() {
        const ua = window.navigator.userAgent;
        const isIos = /iphone|ipad|ipod/i.test(ua);
        const isStandalone = window.navigator.standalone === true;
        return isIos && !isStandalone;
    }

    if (isIosSafari() && iosHint) {
        iosHint.style.display = 'flex';
    }

    if (btnDismissIos) {
        btnDismissIos.addEventListener('click', () => {
            if (iosHint) iosHint.style.display = 'none';
        });
    }

    // ============================================================
    // Run
    // ============================================================
    document.addEventListener('DOMContentLoaded', init);
})();
