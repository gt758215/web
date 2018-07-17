
var colorStatTable = function(setting) {
    var items = $("table.color_state_table tbody td.color_value");

    var max_value = -1;
    var min_value = -1;
    var scale = 0
    var color_rgb = null;
    var main_diagnoal_color_rgb = null;

    var defaults = {
        color : "#f00000",
        main_diagnoal_color: "#00ff00",
        max_opacity: 1.0,
        min_opacity: 0.0
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

    var _get_one_value = function(elm) {

        if (elm = null) return 0;
        item = $(elm);
        if (item.find('a').length != 0) {
            return Number(item.find('a').first().text())
        } else {
            return Number(item.text())
        }
    }

    var _find_max_min = function() {
        var item = $(this);
        var value = 0;
        if (item.find('a').length != 0) {
            value = Number(item.find('a').first().text())
        } else {
            value = Number(item.text())
        }

        if (max_value == -1 || value > max_value) {
            max_value = value;
        }
        if (min_value == -1 || value < min_value ) {
            min_value = value;
        }

    };

    var _init = function() {
        color_rgb = _hexToRgb(options.color);
        console.log('set main diagnoal color to ' + options.main_diagnoal_color );
        if (options.main_diagnoal_color != null) {
            main_diagnoal_color_rgb = _hexToRgb(options.main_diagnoal_color);
        }
        items.each(_find_max_min);
        scale = max_value - min_value;
    }
    var _calcuate_opacity = function(value) {

        if (value == 0 ) {
            return options.min_opacity;
        }

        return options.min_opacity + (value / scale) * (options.max_opacity - options.min_opacity);

    }

    var draw = function() {


        items.each(function(){
            var item = $(this);
            var value = 0;
            if (item.find('a').length != 0) {
                value = Number(item.find('a').first().text())
            } else {
                value = Number(item.text())
            }

            opacity = _calcuate_opacity(value);

            item.css('background-color','rgba(' + color_rgb.r + ', ' + color_rgb.g + ', ' + color_rgb.b + ', ' + opacity + ')');
        });

        if (main_diagnoal_color_rgb != null) {
            rows = $('table.color_state_table tbody tr')
            rows.each(function(i) {
                var row = $(this)
                row.find('td.color_value').each(function (j){
                    console.log("result i:"+ i+", j:"+j);
                    if (i==j) {
                        var block = $(this);
                        var value = 0;
                        if (block.find('a').length != 0) {
                            value = Number(block.find('a').first().text());
                        } else {
                            value = Number(block.text());
                        }
                        opacity = _calcuate_opacity(value);
                        block.css('background-color','rgba(' + main_diagnoal_color_rgb.r + ', ' + main_diagnoal_color_rgb.g + ', ' + main_diagnoal_color_rgb.b + ', ' + opacity + ')');
                        console.log("result op:"+ opacity+", value:"+value+", scale:"+scale);
                    }
                })
            });
        }
    }


    _init()
    return {draw: draw}
};