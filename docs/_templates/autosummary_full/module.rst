{#- the root page is the landing page of the full API toctree, so title it
    accordingly: that title is what shows up in the sidebar / parent toctree. #}
{%- if "." not in fullname %}
{{ "Full API" | underline }}
{%- else %}
{{ fullname | escape | underline }}
{%- endif %}

.. automodule:: {{ fullname }}
   :members:
   :private-members:
   :undoc-members:
   :show-inheritance:
   :ignore-module-all:
{#- ignore-module-all is what makes :private-members: actually bite: without it
    autodoc narrows :members: to __all__, which most nemos modules define. #}

{% block modules %}
{#- all_modules, not modules: the latter drops any submodule whose name starts
    with an underscore, which is most of nemos' internals. #}
{%- if all_modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :template: autosummary_full/module.rst
   :recursive:
{% for item in all_modules %}
   {{ item }}
{%- endfor %}
{%- endif %}
{% endblock %}
