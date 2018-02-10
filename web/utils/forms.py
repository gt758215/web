import wtforms


class Tooltip(object):
    """
    An HTML form tooltip.
    """

    def __init__(self, field_id, for_name, text):
        self.field_id = field_id
        self.text = text
        self.for_name = for_name

    def __str__(self):
        return self()

    def __unicode__(self):
        return self()

    def __html__(self):
        return self()

    def __call__(self, text=None, **kwargs):
        if 'for_' in kwargs:
            kwargs['for'] = kwargs.pop('for_')
        else:
            kwargs.setdefault('for', self.field_id)

        return wtforms.widgets.HTMLString(
            ('<span name="%s_explanation"'
             '    class="explanation-tooltip glyphicon glyphicon-question-sign"'
             '    data-container="body"'
             '    title="%s"'
             '    ></span>') % (self.for_name, self.text))

    def __repr__(self):
        return 'Tooltip(%r, %r, %r)' % (self.field_id, self.for_name, self.text)


class Explanation(object):
    """
    An HTML form explanation.
    """

    def __init__(self, field_id, for_name, filename):
        self.field_id = field_id
        self.file = filename
        self.for_name = for_name

    def __str__(self):
        return self()

    def __unicode__(self):
        return self()

    def __html__(self):
        return self()

    def __call__(self, file=None, **kwargs):
        if 'for_' in kwargs:
            kwargs['for'] = kwargs.pop('for_')
        else:
            kwargs.setdefault('for', self.field_id)

        import flask
        from web.webapp import app

        html = ''
        # get the text from the html file
        with app.app_context():
            html = flask.render_template(file if file else self.file)

        if len(html) == 0:
            return ''

        return wtforms.widgets.HTMLString(
            ('<div id="%s_explanation" style="display:none;">\n'
             '%s'
             '</div>\n'
             '<a href=# onClick="bootbox.alert($(\'#%s_explanation\').html()); '
             'return false;"><span class="glyphicon glyphicon-question-sign"></span></a>\n'
             ) % (self.for_name, html, self.for_name))

    def __repr__(self):
        return 'Explanation(%r, %r, %r)' % (self.field_id, self.for_name, self.file)


class StringField(wtforms.StringField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(StringField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)