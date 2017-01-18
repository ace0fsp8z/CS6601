# Setup
Clone this repository recursively:
`git clone --recursive https://github.gatech.edu/omscs6601/assignment_1.git`

(If your version of git does not support recurse clone, then clone without the option and run `git submodule init` and `git submodule update`).

If you run across certificate authentication issues during the clone, set the git SSL Verify option to false: `git config --global http.sslVerify false`.

## Python Dependencies

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

`git commit -am "<funny message vilifying TAs for the update>"`

# Submit your code
A friendly reminder: please ensure that your submission is in `player_submission.py`. The script described in the following section automatically sends that file to the servers for processing.

To submit your code and have it evaluated for a grade, use `python submit.py assignment_1`. We are going to limit you to 1 submissions in one hour(Subjected to change depending on load on servers) and the last submission before the deadline will be used to determine your grade.

To enter yourself into the playoffs against your classmates, run `python submit.py --enable-face-off assignment_1`. Ensure that you have created the required AI.txt to enter the tournament.

