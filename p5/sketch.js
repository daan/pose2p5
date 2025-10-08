let landmarks = [];

function setup() {
    createCanvas(windowWidth, windowHeight, WEBGL);

    // Connect to the Python SSE stream using EventSource
    const eventSource = new EventSource('/stream');

    // The 'onmessage' event is fired when data is received
    eventSource.onmessage = function(event) {
        landmarks = JSON.parse(event.data);
    };
}

function draw() {
    background(200);
    orbitControl(); // Allows camera movement with the mouse
    
    if (landmarks.length > 0) {
        // Draw spheres for each landmark
        noStroke();
        fill(255, 0, 0);
        for (const lm of landmarks) {
            push();
            // MediaPipe coordinates are normalized, so we scale them.
            // (0,0) is top-left, but in WEBGL it's center. So we subtract 0.5 before scaling.
            let scale = min(width, height);
            let x = (lm.x - 0.5) * scale;
            let y = (lm.y - 0.5) * scale;
            let z = lm.z * scale;
            
            translate(x, y, z);
            sphere(5);
            pop();
        }

        // Draw connections between landmarks
        // This is based on solutions.pose.POSE_CONNECTIONS
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
            [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
            [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [11, 23], [12, 24],
            [23, 24], [23, 25], [25, 27], [27, 29], [29, 31], [24, 26], [26, 28],
            [28, 30], [30, 32]
        ];

        stroke(255);
        strokeWeight(2);
        for (const conn of connections) {
            const lm1 = landmarks[conn[0]];
            const lm2 = landmarks[conn[1]];

            if (lm1 && lm2) {
                let scale = min(width, height);
                let x1 = (lm1.x - 0.5) * scale;
                let y1 = (lm1.y - 0.5) * scale;
                let z1 = lm1.z * scale;
                let x2 = (lm2.x - 0.5) * scale;
                let y2 = (lm2.y - 0.5) * scale;
                let z2 = lm2.z * scale;

                line(x1, y1, z1, x2, y2, z2);
            }
        }
    }
}
