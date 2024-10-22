
function ekUpload() {
    function Init() {
        var fileSelect = document.getElementById("file-upload"),
            fileDrag = document.getElementById("file-drag"),
            submitButton = document.getElementById("submit-button");

        fileSelect.addEventListener("change", fileSelectHandler, false);

        // Is XHR2 available?
        var xhr = new XMLHttpRequest();
        if (xhr.upload) {
            // File Drop
            fileDrag.addEventListener("dragover", fileDragHover, false);
            fileDrag.addEventListener("dragleave", fileDragHover, false);
            fileDrag.addEventListener("drop", fileSelectHandler, false);
        }
    }

    function fileDragHover(e) {
        var fileDrag = document.getElementById("file-drag");

        e.stopPropagation();
        e.preventDefault();

        fileDrag.className = e.type === "dragover" ? "hover" : "modal-body file-upload";
    }

    function fileSelectHandler(e) {
        var files = e.target.files || e.dataTransfer.files;
        fileDragHover(e);
        for (var i = 0, f; (f = files[i]); i++) {
            parseFile(f);
            uploadFile(f);
        }
    }

    function output(msg) {
        var m = document.getElementById("messages");
        m.innerHTML = msg;
    }

    function parseFile(file) {
        output("<strong>" + encodeURI(file.name) + "</strong>");

        var imageName = file.name;

        var isGood = /\.(?=gif|jpg|png|jpeg)/gi.test(imageName);
        if (isGood) {
            document.getElementById("start").classList.add("hidden");
            document.getElementById("response").classList.remove("hidden");
            document.getElementById("notimage").classList.add("hidden");
            document.getElementById("submit-button").style.display = "block";
            document.getElementById("file-image").classList.remove("hidden");
            document.getElementById("file-image").src = URL.createObjectURL(file);
        } else {
            document.getElementById("file-image").classList.add("hidden");
            document.getElementById("notimage").classList.remove("hidden");
            document.getElementById("start").classList.remove("hidden");
            document.getElementById("response").classList.add("hidden");
            document.getElementById("file-upload-form").reset();
        }
    }

    function setProgressMaxValue(e) {
        var pBar = document.getElementById("file-progress");

        if (e.lengthComputable) {
            pBar.max = e.total;
        }
    }

    function updateFileProgress(e) {
        var pBar = document.getElementById("file-progress");

        if (e.lengthComputable) {
            pBar.value = e.loaded;
        }
    }

    function uploadFile(file) {
        var xhr = new XMLHttpRequest(),
            fileInput = document.getElementById("class-roster-file"),
            pBar = document.getElementById("file-progress"),
            fileSizeLimit = 1024; // In MB
        if (xhr.upload) {
            // Check if file is less than x MB
            if (file.size <= fileSizeLimit * 1024 * 1024) {
                // Progress bar
                pBar.style.display = "inline";
                xhr.upload.addEventListener("loadstart", setProgressMaxValue, false);
                xhr.upload.addEventListener("progress", updateFileProgress, false);

                // File received / failed
                xhr.onreadystatechange = function (e) {
                    if (xhr.readyState == 4) {
                        // Everything is good!
                        // progress.className = (xhr.status == 200 ? "success" : "failure");
                        // document.location.reload(true);
                    }
                };

                // Start upload
                xhr.open("POST", document.getElementById("file-upload-form").action, true);
                xhr.setRequestHeader("X-File-Name", file.name);
                xhr.setRequestHeader("X-File-Size", file.size);
                xhr.setRequestHeader("Content-Type", "multipart/form-data");
                // xhr.send(file);
            } else {
                output("Please upload a smaller file (< " + fileSizeLimit + " MB).");
            }
        }
    }

    // Check for the various File API support.
    if (window.File && window.FileList && window.FileReader) {
        Init();
    } else {
        document.getElementById("file-drag").style.display = "none";
    }
}

const handleSubmit = async (e) => {
    e.preventDefault();
    try {
        const file = document.getElementById("file-upload");
        const formData = new FormData();
        formData.append("image", file.files[0]);
        const res = await fetch("/detect", {
            method: "POST",
            body: formData,
        });
        if (res.ok) {
            const imgBlob = await res.blob();
            const imgUrl = URL.createObjectURL(imgBlob)
            const imgElement = document.getElementById("processed-img");
            imgElement.src = imgUrl
            document.getElementById("res_img").style.display = "block";
            document.getElementById("main_form").style.display = "none";
            document.querySelector(".backBtn").style.display = "block";
        }
    } catch (error) {
        alert("an error has occured");
    }
}

ekUpload();
