document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get form values
    const cement = parseFloat(document.getElementById('cement').value);
    const blastFurnaceSlag = parseFloat(document.getElementById('blast-furnace-slag').value);
    const flyAsh = parseFloat(document.getElementById('fly-ash').value);
    const water = parseFloat(document.getElementById('water').value);
    const superplasticizer = parseFloat(document.getElementById('superplasticizer').value);
    const coarseAggregate = parseFloat(document.getElementById('coarse-aggregate').value);
    const fineAggregate = parseFloat(document.getElementById('fine-aggregate').value);
    const age = parseFloat(document.getElementById('age').value);

    // Load the model
    const modelUrl = 'model.json';
    let model;
    try {
        model = await tf.loadLayersModel(modelUrl);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
        document.getElementById('result').innerText = 'Error loading model.';
        return;
    }

    // Create a tensor with the correct shape
    const inputShape = [1, 8];  // Batch size of 1 and 8 features
    const input = tf.tensor2d([[cement, blastFurnaceSlag, flyAsh, water, superplasticizer, coarseAggregate, fineAggregate, age]], inputShape);
    console.log('Input tensor shape:', input.shape);

    // Predict
    let prediction;
    try {
        prediction = model.predict(input);
        const result = prediction.dataSync()[0];
        console.log('Prediction result:', result);
        document.getElementById('result').innerText = `Predicted Compressive Strength: ${result.toFixed(2)} MPa`;
    } catch (error) {
        console.error('Error during prediction:', error);
        document.getElementById('result').innerText = 'Error during prediction.';
    }
});
