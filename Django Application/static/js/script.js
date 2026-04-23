$(document).on("change", "#id_upload_video_file", function () {
    var source = $("#video_source");
    if (!this.files || !this.files[0]) {
        return;
    }

    source[0].src = URL.createObjectURL(this.files[0]);
    source.parent()[0].load();
    $("#videos").css("display", "block");
});

$("form").on("submit", function () {
    $("#videoUpload")
        .prop("disabled", true)
        .html('Running DRISHTI analysis&nbsp;<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>');
});
