(function() {
    'use strict';
  
    // **********************************************
    // Add Intent Button
    // **********************************************

    $('#btnIntentsAddDialogConfirm').on('click', event => {
    event.preventDefault();
    addIntentFunc()
  })

  var addIntentFunc = () => {
    var intentName = $('#intentName').val()
      $.getJSON('/intents/add/' + intentName, res => {
        console.log(res)
        if (res.ok == true) {
            window.location.reload()
        }
        else
        {
            alert(res.message)
        }
    })
    }

  $('#btnIntentsAddDialogCancel').on('click', event => {
    event.preventDefault();

    var $modalEl = $('#AddIntentModalDiv');
    // hide modal
    mui.overlay('off', $modalEl.get(0));

  })
  $('#intentName').keypress(function(e){
    if(e.keyCode==13)
    $("#btnIntentsAddDialogConfirm").trigger('click');
  });

  $('#intentName').select();
  
})();