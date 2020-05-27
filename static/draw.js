
let sketch = (p) => {
    let canvas = null


    p.setup = () => {
        let size = p.min(p.windowWidth * 0.7, p.windowHeight * 0.7)
        canvas = p.createCanvas(size, size);
        p.background('#B7C0EE')
        placeButtons()
    }

    let placeButtons = () => {
        container = p.select('#big')
        // guess button
        guess_btn = p.createButton('guess')
        guess_btn.class('btn btn-lg btn-primary')
        guess_btn.style('margin', '5px')
        container.child(guess_btn)
        guess_btn.mousePressed(guess)

        // clear button
        guess_btn = p.createButton('clear')
        guess_btn.class('btn btn-lg btn-secondary')
        guess_btn.style('margin', '5px')
        container.child(guess_btn)
        guess_btn.mousePressed(clear)
    }

    p.windowResized = () => {
        let size = p.min(p.windowWidth * 0.7, p.windowHeight * 0.7)
        p.resizeCanvas(size, size);
        p.background('#B7C0EE')
    }

    p.draw = () => {
        if (p.mouseIsPressed) {
            p.fill(255);
            p.noStroke()
            p.ellipse(p.mouseX, p.mouseY, 50, 50);

        }

    }

    let clear = () => {
        p.clear()
        p.background('#B7C0EE')
    }

    let guess = () => {
        // p.saveCanvas(canvas, 'number', 'png');
        var imgAsDataURL = canvas.canvas.toDataURL("image/png");

        data = { img: imgAsDataURL }
        fetch(`${window.origin}/guess`, {
            method: "POST",
            credentials: "include",
            body: JSON.stringify(data),
            cache: "no-cache",
            headers: new Headers({
                "content-type": "application/json"
            })
        })
            .then(function (response) {
                if (response.status !== 200) {
                    console.log(`Looks like there was a problem. Status code: ${response.status}`);
                    return;
                }
                response.json().then(function (data) {
                    console.log(data);
                    let subheader = p.select('#subheader')
                    subheader.html(`I think you drew a ${data.guess}`)
                });
            })
            .catch(function (error) {
                console.log("Fetch error: " + error);
            })
    }
}
new p5(sketch, 'sketch_container')


