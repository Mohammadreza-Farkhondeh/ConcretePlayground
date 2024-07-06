document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const cement = parseFloat(document.getElementById('cement').value);
    const blastFurnaceSlag = parseFloat(document.getElementById('blast-furnace-slag').value);
    const flyAsh = parseFloat(document.getElementById('fly-ash').value);
    const water = parseFloat(document.getElementById('water').value);
    const superplasticizer = parseFloat(document.getElementById('superplasticizer').value);
    const coarseAggregate = parseFloat(document.getElementById('coarse-aggregate').value);
    const fineAggregate = parseFloat(document.getElementById('fine-aggregate').value);
    const age = parseFloat(document.getElementById('age').value);

    const model = await tf.loadLayersModel('model/model.json');

    const input = tf.tensor2d([[cement, blastFurnaceSlag, flyAsh, water, superplasticizer, coarseAggregate, fineAggregate, age]]);

    const prediction = model.predict(input);
    const result = prediction.dataSync()[0];

    document.getElementById('result').innerText = `Predicted Compressive Strength: ${result.toFixed(2)} MPa`;
});

