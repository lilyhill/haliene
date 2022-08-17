submit = document.getElementById('submit_button')
submit.addEventListener('click', upload)

function upload() {
    const filePicker = document.querySelector('input');
    if (filePicker.files) {
        file = filePicker.files[0]
        //do something cool with the file
    }
    else {
        console.log("no file chosen bruh")
    }
}


//todo implement function for calling the api
// check how the queing mechanism will work?
// how will we get the results back and display it to the customer
// figure out the progressive web app part