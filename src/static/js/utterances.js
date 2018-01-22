(function() {
    'use strict';
  
    // **********************************************
    // Add utterance Button
    // **********************************************

    function addUtterance() {
        var utteranceBody = $('#utteranceBody').val()
        var intentName = $('#intentName').val()
        var data = {
            utterance_body : utteranceBody,
            intent_name : intentName
        }

        var options = {
            method: 'post',
            contentType: "application/json; charset=utf-8",
            data: JSON.stringify(data),
            dataType: 'json',
            success: res => {
                console.log(res)
                if(res.ok == true)
                {
                    location.reload();
                }
                else
                {
                  alert(res.message)
                }
            }
        }

        $.ajax('/intents/addutterance/', options );
    }

    var $btnAddUtterance = $('#btnAddUtterance');        
    $btnAddUtterance.on('click', addUtterance);

    $('#utteranceBody').keypress(function(e){
        if(e.keyCode==13)
        $("#btnAddUtterance").trigger('click');
      });

    function deleteUtterance() {
        $.getJSON('/intents/deleteutterance/' + $(this).data('id'), res => {
            if(res.ok == true)
            {
                location.reload();
            }
            else
            {
              alert(res.message)
            }
          })
    }
    
    var $btnDeleteUtterance = $('.btnDeleteUtterance');
    $btnDeleteUtterance.on('click', deleteUtterance)

    $('#utteranceBody').select();

})();