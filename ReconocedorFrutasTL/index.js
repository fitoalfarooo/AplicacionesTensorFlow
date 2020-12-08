//By Fabian Alfaro & Warner Cortes

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
let net;

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');

    // Create an object from Tensorflow.js data API which could capture image 
    // from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);

    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = async classId => {
        // Capture an image from the web camera.
        const img = await webcam.capture();

        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(img, true);

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);

        // Dispose the tensor to release the memory.
        img.dispose();
    };


    //https://github.com/tensorflow/tfjs/issues/633

    //Permite guardar el entrenamiento en el formato de Tensores para luego ser utilizado de TL

    const saveModel = async => {

        let dataset = classifier.getClassifierDataset()
        var datasetObj = {}
        Object.keys(dataset).forEach((key) => {
            let data = dataset[key].dataSync();
            datasetObj[key] = Array.from(data);
        });

        var a = document.createElement("a");
        var file = new Blob([JSON.stringify(datasetObj)], { type: 'text/plain' });
        a.href = URL.createObjectURL(file);
        a.download = 'entrenamiento.txt';
        a.click();

    };

    document.getElementById('file-selector')
        .addEventListener('change', function() {

            var fr = new FileReader();
            fr.onload = function() {
                let tensorObj = JSON.parse(fr.result);
                Object.keys(tensorObj).forEach((key) => {
                    tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1024, 1024])
                })
                classifier.setClassifierDataset(tensorObj);
            }

            fr.readAsText(this.files[0]);

        })

    //----------------------------------------------------------------------------------

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
    document.getElementById('save').addEventListener('click', () => saveModel());


    while (true) {
        if (classifier.getNumClasses() > 0) {
            const img = await webcam.capture();

            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(img, 'conv_preds');
            // Get the most likely class and confidence from the classifier module.
            const result = await classifier.predictClass(activation);

            const classes = ['Banano', 'Limon', 'Manzana', 'No se ha reconocido ninguna fruta'];
            document.getElementById('console').innerText = `
        Fruta Reconocida: ${classes[result.label]}\n
        Probabilidad: ${result.confidences[result.label]}
      `;

            // Dispose the tensor to release the memory.
            img.dispose();
        }

        await tf.nextFrame();
    }
}


app();