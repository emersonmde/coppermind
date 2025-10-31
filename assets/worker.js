// CPU-intensive worker for testing parallel computation
self.onmessage = function(e) {
    const { workerId, text, iterations } = e.data;

    console.log(`Worker ${workerId} starting computation`);

    let result = 0;
    let hash = 0;

    // Compute-intensive operations to test worker pool performance
    for (let i = 0; i < iterations; i++) {
        for (let j = 0; j < 50000; j++) {
            result += Math.sqrt(i * j) * Math.sin(i) * Math.cos(j);
            hash ^= Math.floor(Math.tan(i + j) * 1000000);
            result += Math.pow(Math.abs(Math.sin(j)), 0.37);
            result += Math.log(Math.abs(i + j + 1));
            result += Math.atan2(i, j + 1);
        }

        for (let k = 0; k < 1000; k++) {
            result = result * 0.9999999 + Math.exp(k / 10000);
        }
    }

    console.log(`Worker ${workerId} completed`);

    self.postMessage({
        workerId,
        result,
        hash,
        text: text.substring(0, 20)
    });
};
