# How To Login to ALCF[^1] Systems

- Please reach out to [Sam Foreman](mailto://foremans@anl.gov) ([foremans@anl.gov](mailto://foremans@anl.gov)) for any questions or issues related to the content in this section.

## Introduction / Background

- Check out the [Get Started](https://www.alcf.anl.gov/support-center/get-started) page from [ALCF's Support Center](https://www.alcf.anl.gov/support-center) for general guidance on getting started
  - To log into Polaris, you will use the command: `ssh [username]@polaris.alcf.anl.gov`
    - or more generally: `ssh [username]@[system].alcf.anl.gov`
  - When questions or issues arise, feel free to discuss amongst yourselves in the Slack channel or bring the issue up to one of the organizers (we're here to help!)
  - When in doubt, there is a wealth of information available from ALCF on general best practices (+ tips & tricks for troubleshooting) 
  - Some important pages to keep close might be:
    - [ALCF's Support Center](https://www.alcf.anl.gov/support-center)
    - [Get Started](https://www.alcf.anl.gov/support-center/get-started)
    - [Learn to Use Our Systems](https://www.alcf.anl.gov/support-center/get-started/learn-use-our-systems)

- For those who may be unfamiliar with using the command line or even just want a review, there is no shortage of available information to help you get started.
  - [The Linux Command Line for Beginners](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)
  - [The Linux Command Line: A Complete Introduction](https://linuxcommand.org/tlcl.php) (+ complete book, freely available as a PDF)

## Logging in

- Content modified from [Connect and Login](https://www.alcf.anl.gov/support-center/get-started/connect-and-login)

### Linux and macOS

![login gif](img/login_mac.gif)

1. Launch the terminal of your choice
   - **macOS**: `Terminal` (built-in) or [`iTerm`](https://iterm2.com/) (free, modern feature set) are good options
   - **linux**: Virtually all modern linux distributions come pre-installed with an application (`Terminal`, `KDE Konsole`, `XTerm`, etc) to access the command line
   - **References**:
     - [Opening a terminal](https://ubuntu.com/tutorials/command-line-for-beginners#3-opening-a-terminal) from Ubuntu
     - [Using a command line terminal + ssh](https://towardsdatascience.com/a-quick-guide-to-using-command-line-terminal-96815b97b955)

2. To log into an ALCF system, enter the ssh command: 

   ```bash
   ssh [username]@[system].alcf.anl.gov
   ```

3. If using a **mobile** token, enter your four-digit PIN in the mobile token application to generate a passcode. Enter the passcode onscreen.
   - If using a **physical** token, press the button on your CRYPTOCard to generate an eight-digit, one-time passcode. Enter the passcode onscreen, prepending it with your four-digit PIN.

4. Press Enter.

### Windows

- The easiest way to get up and running on Windows is to install Linux
  - [Install WSL | Microsoft Docs](https://docs.microsoft.com/en-us/windows/wsl/install)
  - [Install Ubuntu on Windows 10](https://ubuntu.com/tutorials/ubuntu-on-windows#1-overview)
  - [Get Ubuntu - Microsoft Store](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6?activetab=pivot:overviewtab)
- Once you've successfully installed a Linux distribution of your choice, you can follow the instructions from the [Linux / macOS](#linux-and-macos) section above

[^1]: [Argonne Leadership Computing Facility](https://alcf.anl.gov/)
