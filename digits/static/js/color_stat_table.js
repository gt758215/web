
var colorStatTable = function(selector=null, setting={}) {


    // Init module variables
    var group_selector = (selector != null) ? selector : '.color_state_table';
    var max_value = -1;
    var min_value = -1;
    var scale = 0
    var color_rgb = null;
    var items = null;

    var defaults = {
        color : "#f00000",
        main_diagonal_color: null,
        max_opacity: 1.0,
        min_opacity: 0.0,
        hover_color: '#cccccc',
        max_value: null,
        min_value: null
    }

    var options = $.extend({}, defaults, setting ||{})

    var _hexToRgb = function (hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    var _find_max_min = function() {
        var item = $(this);
        var value = 0;
        if (item.find('a').length != 0) {
            value = parseFloat(item.find('a').first().text())
        } else {
            value = parseFloat(item.text())
        }

        if (max_value == -1 || value > max_value) {
            max_value = value;
        }
        if (min_value == -1 || value < min_value ) {
            min_value = value;
        }

    };

    var _init = function() {
        items = $('td' + group_selector);
        color_rgb = _hexToRgb(options.color);

        items.each(_find_max_min);
        if (options.max_value != null) {
            max_value = options.max_value
        }
        if (options.min_value != null) {
            min_value = options.min_value
        }
        scale = max_value - min_value;

        //console.log("ColorStatTable initialized with selector:" + group_selector + ", max:" + max_value + ", min:" + min_value)
        //console.log("Options => color: " + options.color + ", max_opacity:" + options.max_opacity + ", min_opacity:" + options.min_opacity )
    }

    var _calculate_opacity = function(value) {

        if (value == 0 ) {
            return options.min_opacity;
        }
        var cal_value = (value <= max_value) ? value : max_value;
        return options.min_opacity + ((value - min_value )/ scale) * (options.max_opacity - options.min_opacity);
    }

    var draw = function() {
        items.each(function(){
            var item = $(this);
            var value = 0;
            if (item.find('a').length != 0) {
                value = parseFloat(item.find('a').first().text())
            } else {
                value = parseFloat(item.text())
            }

            opacity = _calculate_opacity(value);

            item.css('background-color','rgba('
                + color_rgb.r + ', '
                + color_rgb.g + ', '
                + color_rgb.b + ', '
                + opacity + ')');
             console.log('set color r:'+color_rgb.r+',g:'+color_rgb.g+',b:'+color_rgb.b+',o:'+opacity);
        });
        items.hover(
            // hover in
            function() {
                $(this).css('background-color', options.hover_color);
            },
            // hover out
            function(){
                var item = $(this);
                var value = 0
                if (item.find('a').length != 0) {
                    value = parseFloat(item.find('a').first().text());
                } else {
                    value = parseFloat(item.text());
                }
                opacity = _calculate_opacity(value);

                item.css('background-color','rgba('
                    + color_rgb.r + ', '
                    + color_rgb.g + ', '
                    + color_rgb.b + ', '
                    + opacity + ')');
            }
        );
    }
    _init()
    return {draw: draw}
};