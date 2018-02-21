import wtforms
from wtforms import SubmitField
from wtforms import validators
from werkzeug.datastructures import FileStorage

def validate_required_iff(**kwargs):
    """
    Used as a validator within a wtforms.Form
    This implements a conditional DataRequired
    Each of the kwargs is a condition that must be met in the form
    Otherwise, no validation is done
    """
    def _validator(form, field):
        all_conditions_met = True
        for key, value in kwargs.iteritems():
            if getattr(form, key).data != value:
                all_conditions_met = False

        if all_conditions_met:
            # Verify that data exists
            if field.data is None \
                    or (isinstance(field.data, (str, unicode)) and not field.data.strip()) \
                    or (isinstance(field.data, FileStorage) and not field.data.filename.strip()):
                raise validators.ValidationError('This field is required.')
        else:
            # This field is not required, ignore other errors
            field.errors[:] = []
            raise validators.StopValidation()

    return _validator


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


class SelectField(wtforms.SelectField):

    def __init__(self, label='', validators=None, tooltip='', explanation_file='', **kwargs):
        super(SelectField, self).__init__(label, validators, **kwargs)

        self.tooltip = Tooltip(self.id, self.short_name, tooltip)
        self.explanation = Explanation(self.id, self.short_name, explanation_file)


def set_data(job, form, key, value):
    if not hasattr(job, 'form_data'):
        job.form_data = dict()
    job.form_data[key] = value

    if isinstance(value, basestring):
        value = '\'' + value + '\''
    return False


def add_warning(form, warning):
    if not hasattr(form, 'warnings'):
        form.warnings = tuple([])
    form.warnings += tuple([warning])
    return True


def iterate_over_form(job, form, function, prefix=['form'], indent=''):

    warnings = False
    if not hasattr(form, '__dict__'):
        return False

    # This is the list of Field types to save. SubmitField and
    # FileField is excluded. SubmitField would cause it to post and
    # FileField can not be populated.
    whitelist_fields = [
        'BooleanField', 'FloatField', 'HiddenField', 'IntegerField',
        'RadioField', 'SelectField', 'SelectMultipleField',
        'StringField', 'TextAreaField', 'TextField',
        'MultiIntegerField', 'MultiFloatField']

    blacklist_fields = ['FileField', 'SubmitField']

    for attr_name in vars(form):
        if attr_name == 'csrf_token' or attr_name == 'flags':
            continue
        attr = getattr(form, attr_name)
        if isinstance(attr, object):
            if isinstance(attr, SubmitField):
                continue
            warnings |= iterate_over_form(job, attr, function, prefix + [attr_name], indent + '    ')
        if hasattr(attr, 'data') and hasattr(attr, 'type'):
            if (isinstance(attr.data, int) or
                isinstance(attr.data, float) or
                isinstance(attr.data, basestring) or
                    attr.type in whitelist_fields):
                key = '%s.%s.data' % ('.'.join(prefix), attr_name)
                warnings |= function(job, attr, key, attr.data)

            # Warn if certain field types are not cloned
            if (len(attr.type) > 5 and attr.type[-5:] == 'Field' and
                attr.type not in whitelist_fields and
                    attr.type not in blacklist_fields):
                warnings |= add_warning(attr, 'Field type, %s, not cloned' % attr.type)
    return warnings


def save_form_to_job(job, form):
    iterate_over_form(job, form, set_data)