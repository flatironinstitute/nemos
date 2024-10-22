# Contributing

The NeMoS package is designed to provide a robust set of statistical analysis tools for neuroscience research. While the repository is managed by a core team of data scientists at the Center for Computational Neuroscience of the Flatiron Institute, we warmly welcome contributions from external collaborators. 
This guide explains how to contribute: if you have questions about the process, please feel free to reach out on [Github Discussions](https://github.com/flatironinstitute/nemos/discussions).

## General Guidelines

Developers are encouraged to contribute to various areas of development. This could include the creation of concrete classes, such as those for new basis function types, or the addition of further checks at evaluation. Enhancements to documentation and the overall readability of the code are also greatly appreciated.

Feel free to work on any section of code that you believe you can improve. More importantly, remember to thoroughly test all your classes and functions, and to provide clear, detailed comments within your code. This not only aids others in using the library, but also facilitates future maintenance and further development.

For more detailed information about NeMoS modules, including design choices and implementation details, visit the [`For Developers`](https://nemos.readthedocs.io/en/latest/developers_notes/) section of the package documentation.

## Contributing to the code

### Contribution workflow cycle

In order to contribute, you will need to do the following:

1) Create your own branch
2) Make sure that tests pass and code coverage is maintained
3) Open a Pull Request

The NeMoS package follows the [Git Flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) workflow. In essence, there are two primary branches, `main` and `development`, to which no one is allowed to
push directly. All development happens in separate feature branches that are then merged into `development` once we have determined they are ready. When enough changes have accumulated, `developemnt` is merged into `main`, and a new release is 
generated. This process includes adding a new tag to increment the version number and uploading the new release to PyPI. 


#### Creating a development environment

You will need a local installation of `nemos` which keeps up-to-date with any changes you make. To do so, you will need to fork and clone `nemos` before checking out
a new branch:

1) Go to the [nemos repo](https://github.com/flatironinstitute/nemos) and click on the `Fork` button at the top right of the page. This will create a copy
of `nemos` in your GitHub account. You should then clone *your fork* to your local machine.

```bash
git clone https://github.com/<YourUserName>/nemos.git
cd nemos
```

2) Install `nemos` in editable mode with developer dependencies  

```bash
pip install -e .[dev]
```

> [!NOTE]
> In order to install `nemos` in editable mode you will need a Python virtual environment. Please see our documentation [here](https://nemos.readthedocs.io/en/latest/installation/) that provides guidance on how to create and activate a virtual environment.

3) Add the upstream branch:

```bash
git remote add upstream https://github.com/flatironinstitute/nemos
```

At this point you have two remotes: `origin` (your fork) and `upstream` (the canonical version). You won't have permission to push to upstream (only `origin`), but 
this make it easy to keep your `nemos` up-to-date with the canonical version by pulling from upstream: `git pull upstream`.

#### Creating a new branch

As mentioned previously, each feature in `nemos` is worked on in a separate branch. This allows multiple people developing multiple features simultaneously, without interfering with each other's work. To create
your own branch, run the following from within your `nemos` directory:

> [!NOTE]
> Below we are checking out the `development` branch. In terms of the `nemos` contribution workflow cycle, the `development` branch accumulates a series of changes from different feature branches that are then all merged into the `main` branch at one time (normally at the time of a release).

```bash
# switch to the development branch on your local copy
git checkout development
# update your local copy from your fork
git pull origin development
# sync your local copy with upstream development
git pull upstream development
# update your fork's development branch with any changs from upstream
git push origin development
# create and switch to a new branch, where you'll work on your new feature
git checkout -b my_feature_branch
```

After you have made changes on this branch, add and commit them when you are ready:

```bash
# stage the changes
git add src/nemos/the_changed_file.py
# commit your changes
git commit -m "A one-line message explaining the changes made"
# push to the remote origin
git push origin my_feature_branch
```

#### Contributing your change back to NeMoS

You can make any number of changes on your branch. Once you are happy with your changes, add tests to check that they run correctly and add documentation to properly note your changes. 
See below for details on how to [add tests](#adding-tests) and properly [document](#adding-documentation) your code.

Lastly, you should make sure that the existing tests all run successfully and that the codebase is formatted properly:

> [!TIP]
> The [NeMoS GitHub action](.github/workflows/ci.yml) runs tests and some additional style checks in an isolated environment using [`tox`](https://tox.wiki/en/). `tox` is not included in our optional dependencies, so if you want to replicate the action workflow locally, you need to install `tox` via pip and then run it. From the package directory:
> ```sh
> pip install tox
> tox -e check,py
> ```
> This will execute `tox` with a Python version that matches your local environment. If the above passes, then the Github action will pass and your PR is mergeable
>
> You can also use `tox` to use `black` and `isort` to try and fix your code if either of those are failing. To do so, run `tox -e fix`
> 
> `tox` configurations can be found in the [`tox.ini`](tox.ini) file.


```bash
# run tests and make sure they all pass
pytest tests/

# run doctest (run all examples in docstrings and match output)
pytest --doctest-modules src/nemos/ 

# format the code base
black src/
isort src --profile=black
isort docs/how_to_guide --profile=black
isort docs/background --profile=black
isort docs/tutorials --profile=black
flake8 --config=tox.ini src
```

> [!IMPORTANT] 
> [`black`](https://black.readthedocs.io/en/stable/) and [`isort`](https://pycqa.github.io/isort/) automatically reformat your code and organize your imports, respectively. [`flake8`](https://flake8.pycqa.org/en/stable/#) does not modify your code directly; instead, it identifies syntax errors and code complexity issues that need to be addressed manually.

> [!NOTE]
> If some files were reformatted after running `black`, make sure to commit those changes and push them to your feature branch as well. 

Now you are ready to make a Pull Request (PR). You can open a pull request by clicking on the big `Compare & pull request` button that appears at the top of the `nemos` repo 
after pushing to your branch (see [here](https://intersect-training.org/collaborative-git/03-pr/index.html) for a tutorial).

Your pull request should include the following:
- A summary including information on what you changed and why
- References to relevant issues or discussions
- Special notice to any portion of your changes where you have lingering questions (e.g., "was this the right way to implement this?") or want reviewers to pay special attention to

Next, we will be notified of the pull request and will read it over. We will try to give an initial response quickly, and then do a longer in-depth review, at which point 
you will probably need to respond to our comments, making changes as appropriate. We'll then respond again, and proceed in an iterative fashion until everyone is happy with the proposed 
changes. 

Additionally, every PR to `main` or `development` will automatically run linters and tests through a [GitHub action](https://docs.github.com/en/actions). Merges can happen only when all check passes.

Once your changes are integrated, you will be added as a GitHub contributor and as one of the authors of the package. Thank you for being part of `nemos`!

### Style Guide

The next section will talk about the style of your code and specific requirements for certain feature development in `nemos`. 

- Longer, descriptive names are preferred (e.g., x is not an appropriate name for a variable), especially for anything user-facing, such as methods, attributes, or arguments.
- Any public method or function must have a complete type-annotated docstring (see below for details). Hidden ones do not need to have complete docstrings, but they probably should.

### Releases

We create releases on Github, deploy on / distribute via [pypi](https://pypi.org/), and try to follow [semantic versioning](https://semver.org/):

> Given a version number MAJOR.MINOR.PATCH, increment the:
> 1. MAJOR version when you make incompatible API changes
> 2. MINOR version when you add functionality in a backward compatible manner
> 3. PATCH version when you make backward compatible bug fixes

ro release a new version, we [create a Github release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) with a new tag incrementing the version as described above. Creating the Github release will trigger the deployment to pypi, via our `deploy` action (found in `.github/workflows/deploy-pure-python.yml`). The built version will grab the version tag from the Github release, using [setuptools_scm](https://github.com/pypa/setuptools_scm).

### Testing 

To run all tests, run `pytest` from within the main `nemos` repository. This may take a while as there are many tests, broken into several categories. 
There are several options for how to run a subset of tests:
- Run tests from one file: `pytest tests/test_glm.py`
- Run a specific test within a specific module: `pytests tests/test_glm.py::test_func`
- Another example specifying a test method via the command line: `pytest tests/test_glm.py::GLMClass::test_func`

#### Adding tests

New tests can be added in any of the existing `tests/test_*.py` scripts. Tests should be functions, contained within classes. The class contains a bunch of related tests
(e.g., regularizers, bases), and each test should ideally be a unit test, only testing one thing. The classes should be named `TestSomething`, while test functions should be named 
`test_something` in snakecase.

If you're adding a substantial bunch of tests that are separate from the existing ones, you can create a new test script. Its name must begin with `test_`, 
it must have an `.py` extension, and it must be contained within the `tests` directory. Assuming you do that, our github actions will automatically find it and 
add it to the tests-to-run.

> [!NOTE]
> If you have many variants on a test you wish to run, you should make use of pytest's `parameterize` mark. See the official documentation [here](https://docs.pytest.org/en/stable/how-to/parametrize.html) and NeMoS [`test_error_invalid_entry`](https://github.com/flatironinstitute/nemos/blob/main/tests/test_vallidation.py#L27) for a concrete implementation.

> [!NOTE]
> If you are using an object that gets used in multiple tests (such as a model with certain data, regularizer, or solver), you should use pytest's `fixtures` to avoid having to load or instantiate the object multiple times. Look at our `conftest.py` to see already available fixtures for your tests. See the official documentation [here](https://docs.pytest.org/en/stable/how-to/fixtures.html).

### Documentation 

Documentation is a crucial part of open-source software and greatly influences the ability to use a codebase. As such, it is imperative that any new changes are
properly documented as outlined below. 

#### Adding documentation

1. **Docstrings**

    All public-facing functions and classes should have complete docstrings, which start with a one-line short summary of the function, a medium-length description of the function/class and what it does, a complete description of all arguments and return values, and an example to illustrate usage. Math should be included in a `Notes` section when necessary to explain what the function is doing, and references to primary literature should be included in a `References` section when appropriate. Docstrings should be relatively short, providing the information necessary for a user to use the code.
    
    Private functions and classes should have sufficient explanation that other developers know what the function/class does and how to use it, but do not need to be as extensive.
    
    We follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) conventions for docstring structure.

2. **Examples/Tutorials**

    If your changes are significant (add a new functionality or drastically change the current codebase), then the current examples may need to be updated or a new example may need to be added.
    
    All examples live within the `docs/` subfolder of `nemos`. These are written as `.py` files but are converted to notebooks by [`mkdocs-gallery`](https://smarie.github.io/mkdocs-gallery/), and have a special syntax, as demonstrated in this [example gallery](https://smarie.github.io/mkdocs-gallery/generated/gallery/).
    
    We avoid using `.ipynb` notebooks directly because their JSON-based format makes them difficult to read, interpret, and resolve merge conflicts in version control.
    
    To see if changes you have made break the current documentation, you can build the documentation locally.
    
    ```
    # Clear the cached documentation pages
    # This step is only necessary if your changes affected the src/ directory
    rm -r docs/generated
    # build the docs within the nemos repo
    mkdocs build
    ```
    
    If the build fails, you will see line-specific errors that prompted the failure.

3. **Doctest: Test the example code in your docs**

    Doctests are a great way to ensure that code examples in your documentation remain accurate as the codebase evolves. With doctests, we will test any docstrings, Markdown files, or any other text-based documentation that contains code formatted as interactive Python sessions.
    
    - **Docstrings:**
        To include doctests in your function and class docstrings you must add an `Examples` section. The examples should be formatted as if you were typing them into a Python interactive session, with `>>>` used to indicate commands and expected outputs listed immediately below.
        
        ```python
        def add(a, b):
            """
            The sum of two numbers.
                
            ...Other docstrings sections (Parameters, Returns...)
                
            Examples
            --------
            An expected output is required.
            >>> add(1, 2)
            3
            
            Unless the output is captured.
            >>> out = add(1, 2)
            
            """
            return a + b
        ```
      
        To validate all your docstrings examples, run pytest `--doctest-module` flag,
        
        ```
        pytest --doctest-modules src/nemos/
        ```
      
        This test is part of the Continuous Integration, every example must pass before we can merge a PR.
    
    - **Documentation Pages:**
        Doctests can also be included in Markdown files by using code blocks with the `python` language identifier and interactive Python examples. To enable this functionality, ensure that code blocks follow the standard Python doctest format:
        
        ```markdown
           ```python
           >>> # Add any code
           >>> x = 3 ** 2
           >>> x + 1
           10
      
           ```
        ```
      
        To run doctests on a text file, use the following command:
        
        ```
        python -m doctest -v path-to-your-text-file/file_name.md
        ```
        
        All MarkDown files will be tested as part of the Continuous Integration.

> [!NOTE]
> All internal links to NeMoS documentation pages **must be relative**. Using absolute links can lead to broken references whenever the documentation structure is altered. Any presence of absolute links to documentation pages will cause the continuous integration checks to fail. Please ensure all links follow the relative format before submitting your PR.
