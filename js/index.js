function softmax(logits) {
    // Trouve le maximum pour la stabilité numérique (évite overflow)
    const maxLogit = Math.max(...logits);
    
    // Calcule exp(x - max) pour chaque élément
    const exps = logits.map(x => Math.exp(x - maxLogit));
    
    // Somme des exponentielles
    const sumExps = exps.reduce((a, b) => a + b, 0);
    
    // Normalise pour obtenir les probabilités
    return exps.map(exp => exp / sumExps);
}

async function preprocessCanvas(canvas) {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    const targetWidth = 28;
    const targetHeight = 28;
    tempCanvas.width = targetWidth;
    tempCanvas.height = targetHeight;
    
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, targetWidth, targetHeight);
    tempCtx.drawImage(canvas, 0, 0, targetWidth, targetHeight);
    
    const imageData = tempCtx.getImageData(0, 0, targetWidth, targetHeight);
    const { data } = imageData;
    
    const float32Data = new Float32Array(targetWidth * targetHeight);
    
    // Calcul des statistiques pour debug
    let sum = 0;
    let nonZeroCount = 0;
    
    for (let i = 0; i < targetWidth * targetHeight; i++) {
        const avg = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3;
        float32Data[i] = avg / 255.0;
        sum += float32Data[i];
        if (float32Data[i] > 0.01) nonZeroCount++;
    }
    
    const mean = sum / (targetWidth * targetHeight);
    console.log('=== Stats des données d\'entrée ===');
    console.log('Moyenne des pixels:', mean.toFixed(4));
    console.log('Pixels non-noirs (>0.01):', nonZeroCount);
    console.log('Min:', Math.min(...float32Data).toFixed(4));
    console.log('Max:', Math.max(...float32Data).toFixed(4));
    
    return new ort.Tensor('float32', float32Data, [1, 1, targetHeight, targetWidth]);
}

async function classifyDrawing(session, canvas) {
    try {
        const inputTensor = await preprocessCanvas(canvas);
        
        console.log('Input tensor shape:', inputTensor.dims);
        
        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;
        
        const results = await session.run(feeds);
        
        const output = results[session.outputNames[0]];

        const logits = Array.from(output.data);
        const probabilities = softmax(logits);
        
        console.log('=== Résultats du modèle ===');
        console.log('Logits bruts:', logits.map(x => x.toFixed(2)));
        console.log('Logits min/max:', Math.min(...logits).toFixed(2), '/', Math.max(...logits).toFixed(2));
        console.log('Probabilités:', probabilities.map(x => (x * 100).toFixed(2) + '%'));
        console.log('Somme des probas:', probabilities.reduce((a, b) => a + b, 0).toFixed(6));
        
        let maxProb = -Infinity;
        let predictedDigit = -1;
        
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedDigit = i;
            }
        }
        
        return { digit: predictedDigit, confidence: maxProb, allProbabilities: probabilities };
        
    } catch (e) {
        console.error('Erreur complète:', e);
        throw new Error(`Erreur lors de la classification: ${e}`);
    }
}

function setupDrawingCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    
    // Style du canvas
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20;  // Augmenté pour mieux correspondre à MNIST
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Événements de dessin
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        ctx.beginPath();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
    });
    
    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
    });
    
    canvas.addEventListener('mouseleave', () => {
        isDrawing = false;
    });
    
    // Support tactile
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        ctx.beginPath();
        ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
    });
    
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
        ctx.stroke();
    });
    
    canvas.addEventListener('touchend', () => {
        isDrawing = false;
    });
}

async function main() {
    try {
        document.body.innerHTML = '<div class="container"><h2>Classification de chiffres dessinés</h2><div class="loading-message">Chargement du modèle...</div></div>';
        
        const session = await ort.InferenceSession.create('./CNNmodel.onnx');
        
        document.querySelector('.loading-message').textContent = 'Modèle chargé avec succès!';
        
        
        setTimeout(() => {
            const container = document.querySelector('.container');
            container.innerHTML = `
                <h2>Classification de chiffres dessinés</h2>
                <canvas id="drawingCanvas" width="280" height="280"></canvas>
                <div>
                    <button id="clearBtn">Effacer</button>
                    <button id="predictBtn">Reconnaître le chiffre</button>
                </div>
                <div id="results"></div>
            `;
            
            setupDrawingCanvas();
            
            document.getElementById('clearBtn').addEventListener('click', () => {
                const canvas = document.getElementById('drawingCanvas');
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                document.getElementById('results').innerHTML = '';
            });
            
            document.getElementById('predictBtn').addEventListener('click', async () => {
                const canvas = document.getElementById('drawingCanvas');
                document.getElementById('results').innerHTML = 'Analyse en cours...';
                
                try {
                    const result = await classifyDrawing(session, canvas);
                    
                    let html = `<h3>Chiffre reconnu: ${result.digit}</h3>`;
                    html += `<p>Confiance: ${(result.confidence * 100).toFixed(2)}%</p>`;
                    html += '<h4>Toutes les probabilités:</h4>';
                    
                    result.allProbabilities.forEach((prob, i) => {
                        const percentage = (prob * 100).toFixed(2);
                        html += `${i}: ${percentage}%<br>`;
                    });
                    
                    document.getElementById('results').innerHTML = html;
                } catch (e) {
                    document.getElementById('results').innerHTML = `❌ Erreur: ${e.message}`;
                }
            });
        }, 1000);

    } catch (e) {
        console.error('Erreur complète:', e);
        document.body.innerHTML = `<div class="container"><h2>Erreur</h2><p style="color: red;">❌ Erreur: ${e}<br>Stack: ${e.stack}</p></div>`;
    }
}

main();