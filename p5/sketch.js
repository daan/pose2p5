let bodyPose = { x: 0, y: 0, z: 0 };

function setup() {
    createCanvas(400, 400, WEBGL);

    // Connect to the Python SSE stream using EventSource
    const eventSource = new EventSource('http://localhost:5000/stream');

    // The 'onmessage' event is fired when data is received
    eventSource.onmessage = function(event) {
        bodyPose = JSON.parse(event.data);
    };
}

function draw() {
    background(220);
    translate(bodyPose.x * width - width / 2, bodyPose.y * height - height / 2, bodyPose.z * 100);
    box(50);
}