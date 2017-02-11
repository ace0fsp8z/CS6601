# CS 6601: Artificial Intelligence - Assignment 2, Search

Find the assignment description [here](https://docs.google.com/document/d/1Fct-45PuiT500cYda_15SpWBA44iBObLXQAhuTsWcTs/pub).
Please also read the [FAQ](https://github.gatech.edu/omscs6601/assignment_2/blob/master/FAQ.txt).

Here are some notes you might find useful.
<ol>
  <li><a href="https://docs.google.com/document/d/14Wr2SeRKDXFGdD-qNrBpXjW8INCGIfiAoJ0UkZaLWto/pub" target="_blank">Bi-directional search</a></li>
  <li><a href="https://docs.google.com/document/d/1YEptGbSYUtu180MfvmrmA4B6X9ImdI4oOmLaaMRHiCA/pub" target="_blank">Using Landmarks</a></li>
</ol>

# Setup
Clone this repository recursively:
`git clone --recursive https://github.gatech.edu/omscs6601/assignment_2.git`

(If your version of git does not support recurse clone, then clone without the option and run `git submodule init` and `git submodule update`).

If you run across certificate authentication issues during the clone, set the git SSL Verify option to false: `git config --global http.sslVerify false`.

# Python Dependencies

The submission scripts depend on the presence of 2 python packages - `requests` and `future`. If you are missing either of these packages, install them from the online Python registries. The easiest way to do this is through pip:

`pip install requests future`

# Keeping your code upto date
After the clone, we recommend creating a branch and developing your agents on that branch:

`git checkout -b develop`

(assuming develop is the name of your branch)

Should the TAs need to push out an update to the assignment, commit (or stash if you are more comfortable with git) the changes that are unsaved in your repository:

`git commit -am "<some funny message>"`

Then update the master branch from remote:

`git pull origin master`

This updates your local copy of the master branch. Now try to merge the master branch into your development branch:

`git merge master`

(assuming that you are on your development branch)

There are likely to be merge conflicts during this step. If so, first check what files are in conflict:

`git status`

The files in conflict are the ones that are "Not staged for commit". Open these files using your favourite editor and look for lines containing `<<<<` and `>>>>`. Resolve conflicts as seems best (ask a TA if you are confused!) and then save the file. Once you have resolved all conflicts, stage the files that were in conflict:

`git add -A .`

Finally, commit the new updates to your branch and continue developing:

`git commit -am "<funny message vilifying Bonnie>"`

# Submit your code
To submit your code to have it evaluated for a grade, use `python submit.py assignment_2`.  You may submit as many times as you like.  The last submission before the deadline will be used to determine your grade.

To add a data.pickle file to your submission (containing landmarks of the Atlanta map for improved tri-directional/custom_search), use `python submit.py assignment_2 --add-data`.

A friendly reminder: please ensure that your submission is in `search_submission.py`. The submit script described automatically sends that file to the servers for processing.

# Vagrant

You have the option of using vagrant to make sure that your local code runs in the same environment as the servers on Bonnie (make sure you have [Vagrant](https://www.vagrantup.com/) and [Virtualbox](https://www.virtualbox.org/wiki/Downloads) installed).  To use this option run the following commands in the root directory of your assignment:

```
vagrant up --provider virtualbox
vagrant ssh
```

Your code lives in the `/vagrant` folder within this virtual machine. Changes made to files in your assignment folder will automatically be reflected within the machine.

# Azure Notebooks

Azure has a service for creating and hosting your iPython notebooks. Find it [here](https://notebooks.azure.com/). You can even use your Georgia Tech credentials to sign in. 
