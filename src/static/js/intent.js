(function() {
    'use strict';
  
    // **********************************************
    // Add Intent Button
    // **********************************************

    function addIntent() {
        // initialize modal element
        var $modalEl = $('<div id="AddIntentModalDiv"/>');
    
        // set style
        $modalEl.css({
          width: 400,
          margin: '100px auto',
          backgroundColor: '#fff'
        });
    
        // add content
        $.get("/intents/addIntentDialogBox", data => {
            $modalEl.append(data);
        });
    
        // show modal
        mui.overlay('on', $modalEl.get(0));
    }
    var $btnAddIntent = $('#btnAddIntent');        
    $btnAddIntent.on('click', addIntent);

    function detailIntent() {
        window.location.href = '/intents/detail/' + $(this).data('id')
        return false
    }
    
    var $divIntent = $('.divIntent');
    $divIntent.on('click', detailIntent)

    function deleteIntent() {
        $.getJSON('/intents/delete/' + $(this).data('id'), res => {
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
    
    var $btnDeleteIntent = $('.btnDeleteIntent');
    $btnDeleteIntent.on('click', deleteIntent)



})();