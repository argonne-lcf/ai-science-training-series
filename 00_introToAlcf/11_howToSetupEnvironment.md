# How to setup your environment

After you login to Theta, by default you will land in a [Bash](https://www.gnu.org/software/bash/) interactive shell session. Understanding the basics of this command line interface is essential to effectively performing your science on the remote machine. 

The term _environment_ refers to the software that is easily accessible to you via the command line. For instance, if I type a command `ls ./` to list the local directory contents, the actual executable `ls` lives somewhere and Bash needs to know where to find it. In order to tell Bash where to find executables, you add paths to a Bash environment variable called `PATH`. For instance when you login to Theta you can type `echo $PATH` to see all the default places Bash has been told by our system administrators to look for executables. You can type `which ls` to find out where Bash has found the command `ls`. 

# Intro to Modules

![module gif](img/module_usage.gif)

Theta uses [Modules](https://modules.readthedocs.io/en/latest/) to control the loading of software environments. When you load a module, Bash environment variables are altered to add or remove software executables and/or libraries from your terminal's search space. This doesn't "install" software or change where the software is located, it simply tells Bash where to find the software so you can use it.

Let's take Python as an example. If I want to run a Python script, I need to type `python script_name.py`. However, Bash needs to know where to find the `python` executable. If you have just logged into Theta and type `which python` you might notice Bash finds it here `/usr/bin/python` and running `python --version` tells you that this is Python v2.7.18. We want to use a more modern Python for our example so we will run:
```bash
module load conda/2021-09-22
```
This is a Python distribution installed in September 2021, so it's up to date. If you run `which python` again, you'll find the Python executable is now something similar to `/lus/theta-fs0/software/datascience/conda/conda/2021-09-22/mconda3/bin/python` and the version is v3.8.10. There are many modules installed on Theta, but we will focus on using this Python environment.

It is important to know that when you login to any supercomputer, you land on a _login_ node which typically has standard CPUs on it while the _worker_ nodes will have a different configuration, including the high-speed network. This means software must be compiled carefully to run on the _worker_ node. This may mean that if you compile code for the _worker_ node, you may also not be able to run it on the _login_ node without getting hard to interpret errors. Therefore, if you need to compile software, we recommend doing so on the _worker_ node. You can do this using an _interactive_ job which will be covered in the job submission section.

## Module commands
* `module list`: list currently loaded modules
* `module avail`: list modules available to be loaded
* `module load <module-name>`: load a module
* `module unload <module-name>`: unload a module

# Editors

When you work on a remote system, editing code requires either an editor that runs in your bash terminal or that operates remotely. We'll focus on editing via the bash terminal here.

There are two primary text editors you'll be (almost) guaranteed to find on every supercomputer: [Emacs](https://www.gnu.org/software/emacs/tour/) and [Vi](https://www.guru99.com/the-vi-editor.html). They offer similar capabilities with different keyboard interfaces. You can run these editors by typing their command names `emacs` or `vi` plus a filename (new or existing): 
* `vi filename.py`
* `emacs filename.py`

There are many keyboard reference "cheatsheets" online for both editors, just run a search for them:
* [Emacs Cheatsheet](https://www.gnu.org/software/emacs/refcards/pdf/refcard.pdf)
* [Vi Cheatsheet](http://www.atmos.albany.edu/daes/atmclasses/atm350/vi_cheat_sheet.pdf)

These two editors are highly configurable, and although they use a simple text-based interface by default, they can aso process mouse input if configured properly. If you save the following code to `$HOME/.vimrc` it will configure your `vi` editor in reasonable ways. 
```vim
" --------------------------------------------------------------------------
" To get additional information / help about any of the options below,
" type :help option-name, for example to get help about syntax highlighting
" type :help syntax and press enter (<CR>)
" --------------------------------------------------------------------------

" enable syntax highlighting
syntax on

" enable mouse
set mouse=a
if has("mouse_sgr")
   set ttymouse=sgr
else
   set ttymouse=xterm2
end
" set autoindent
set ai
" number of spaces that a <Tab> in the file counts for 
set tabstop=4
" Allow specified keys that move the cursor left/right to move to
" the previous/next line when the cursor is on the first/last character
" in the line
set whichwrap+=<,>,h,l,[,]
" In Insert mode: Use the appropriate number of space to insert a <Tab>
set expandtab
" How many spaces are equivalent to a single <Tab>
" this is a matter of personal preference, the general recommendation
" for `.py` files is 4, as per pep8
"   - https://www.python.org/dev/peps/pep-0008/#maximum-line-length
set shiftwidth=3
" highlight search results
set hlsearch
" Ignore case when searching
set ignorecase
" Unless capital letters are explicitly included
set smartcase
" Incrementally search through buffer
set incsearch
" show the line and column number of the cursor position
set ruler
" highlight the line currently occupied by the cursor 
set cursorline
" set the title of the window to 'titlestring' if not empty,
" otherwise use filename
set title
" when a bracket is inserted, briefly jump to the matching one
set showmatch
" string to put at the start of lines that have been wrapped
set showbreak=\ \ \
xnoremap p pgvy
set backspace=indent,eol,start

" Uncomment the following to have Vim jump to the last position when
" reopening a file
if has("autocmd")
  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif
```

![vi gif](img/vi_usage.gif)

If you save the following to your `$HOME/.emacs.el` configuration file, it will configure your Emacs editor in reasonable ways:
```
;; Do not display Welcome buffer on GUI init
(setq inhibit-startup-screen t)

;; Automatically remove trailing whitespace when file is saved.
(add-hook 'text-mode-hook
      (lambda()
        (add-hook 'local-write-file-hooks
              '(lambda()
                 (save-excursion
                   (delete-trailing-whitespace))))))

(setq-default indent-tabs-mode nil)
(setq-default tab-width 4)
;; Set the number to the number of columns to use.
(setq-default fill-column 90)

;; Show line number in the mode line.
(line-number-mode 1)

;; Show column number in the mode line.
(setq column-number-mode t)

;; Put line numbers on left column
(global-linum-mode 1)

(load-library "python")

(autoload 'python-mode "python-mode" "Python Mode." t)
(add-to-list 'auto-mode-alist '("\\.py\\'" . python-mode))
(add-to-list 'interpreter-mode-alist '("python" . python-mode))
(add-hook 'python-mode-hook
      (lambda()
        (add-hook 'local-write-file-hooks
              '(lambda()
                 (save-excursion
                   (delete-trailing-whitespace))))))

```
## Remote Development with Visual Studio Code

VS Code is a popular, free GUI editor that works on Linux, macOS, and Windows. To use VS Code on your local machine to edit files on remote machines like ThetaGPU, install the [Remote Development Extension](https://www.alcf.anl.gov/news/argonne-augments-theta-supercomputer-gpus-accelerate-coronavirus-research) and connect to the remote host via [these instructions](https://code.visualstudio.com/docs/remote/ssh). 

# Git Repo
You're going to want to check out the GitHub repo for this tutorial which contains files you can run, etc.
```bash
git clone https://github.com/argonne-lcf/ai-science-training-series.git
```
This will create a folder `ai-science-training-series` where you will find all the contents of the repo. You will want to run `git pull` at the start of each tutorial session to grab the latest updates from our team.

