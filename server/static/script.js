let resultsChartInstance = null;
let currentEventSource = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchAggregatedResults();

    const dropZone = document.getElementById('drop-zone');
    const audioInput = document.getElementById('audio-input');
    const trainForm = document.getElementById('train-form');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const terminalWrapper = document.getElementById('terminal-wrapper');
    const terminalBody = document.getElementById('terminal-body');

    // Train Form Handler
    trainForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const dataset = document.getElementById('dataset-select').value;
        const model = document.getElementById('model-select').value;
        
        // UI Toggles
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        terminalWrapper.classList.remove('hidden');
        terminalBody.innerHTML = '<i>Initializing pipeline backend...</i><br>';
        
        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, model })
            });
            const data = await res.json();
            
            if(data.status === 'started') {
                startStreamingLogs(data.run_id);
            } else {
                terminalBody.innerHTML += `<span style="color: red;">Error starting: ${data.message}</span>`;
            }
        } catch(e) {
            console.error(e);
            terminalBody.innerHTML += `<span style="color: red;">Network Error during pipeline launch.</span>`;
            resetTrainingUI();
        }
    });

    stopBtn.addEventListener('click', () => {
        if(currentEventSource) {
            currentEventSource.close();
        }
        terminalBody.innerHTML += `<br><span style="color: orange;">--- [FORCE STOPPED BY USER] ---</span>`;
        resetTrainingUI();
    });

    function startStreamingLogs(runId) {
        if(currentEventSource) currentEventSource.close();
        
        currentEventSource = new EventSource(`/api/stream/${runId}`);
        
        currentEventSource.onmessage = function(event) {
            // Append log line
            const textNode = document.createTextNode(event.data);
            const br = document.createElement('br');
            
            // Format some lines
            let textHtml = event.data;
            if (textHtml.includes('ERROR') || textHtml.includes('FAILED')) {
                textHtml = `<span style="color: red;">${textHtml}</span>`;
            } else if (textHtml.includes('COMPLETED SUCCESSFULLY')) {
                textHtml = `<span style="color: #27c93f;">${textHtml}</span>`;
            }
            
            terminalBody.innerHTML += textHtml + '<br>';
            
            // Auto-scroll to bottom
            terminalBody.scrollTop = terminalBody.scrollHeight;
            
            // If run completed, close the connection
            if(event.data.includes('RUN COMPLETED SUCCESSFULLY') || event.data.includes('RUN FAILED')) {
                currentEventSource.close();
                resetTrainingUI();
                fetchAggregatedResults(); // Refresh charts
            }
        };

        currentEventSource.onerror = function(err) {
            console.error("EventSource failed:", err);
            terminalBody.innerHTML += `<br><span style="color: red;">[Connection to stream lost]</span>`;
            currentEventSource.close();
            resetTrainingUI();
        };
    }

    function resetTrainingUI() {
        startBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
    }

    // Drag & Drop Handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });
});

async function handleFileUpload(file) {
    if (!file.name.endsWith('.wav')) {
        alert("Please upload a .wav file.");
        return;
    }

    const formData = new FormData();
    formData.append('audio', file);
    
    // Add selected model for inference
    const selectedModel = document.getElementById('model-select').value;
    formData.append('model', selectedModel);

    const loader = document.getElementById('loader');
    const resultBox = document.getElementById('result-box');
    
    loader.classList.remove('hidden');
    resultBox.classList.add('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("API Request Failed");

        const data = await response.json();
        
        document.getElementById('res-model_used').innerText = data.model.split('/').pop();
        document.getElementById('res-emotion').innerText = data.predicted_emotion;
        document.getElementById('res-confidence').innerText = `${Math.round(data.confidence * 100)}%`;
        
        loader.classList.add('hidden');
        resultBox.classList.remove('hidden');

    } catch (error) {
        console.error(error);
        alert("An error occurred during prediction.");
        loader.classList.add('hidden');
    }
}

async function fetchAggregatedResults() {
    try {
        const response = await fetch('/api/results');
        if (!response.ok) return;

        const data = await response.json();
        
        document.getElementById('res-w2v2').innerText = data.best_layers.wav2vec2;
        document.getElementById('res-hubert').innerText = data.best_layers.HuBERT;
        document.getElementById('res-whisper').innerText = data.best_layers.Whisper;
        document.getElementById('res-wavlm').innerText = data.best_layers.WavLM;
        document.getElementById('res-mert').innerText = data.best_layers.MERT;
        document.getElementById('res-w2v_bert').innerText = data.best_layers.w2v_bert;
        
        renderChart(data.chart_data);
    } catch (error) {
        console.error("Failed to load pre-computed stats", error);
    }
}

function renderChart(chartData) {
    const ctx = document.getElementById('resultsChart').getContext('2d');
    
    if(resultsChartInstance) {
        resultsChartInstance.destroy();
    }
    
    resultsChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels, 
            datasets: [
                {
                    label: 'wav2vec 2.0',
                    data: chartData.wav2vec2,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'HuBERT',
                    data: chartData.hubert,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'Whisper',
                    data: chartData.whisper,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'WavLM',
                    data: chartData.wavlm,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'MERT',
                    data: chartData.mert,
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'w2v-BERT',
                    data: chartData.w2v_bert,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}
