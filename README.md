# JANNE (JUNO Adversorial Neural Network Experiment)

This project aim to give a informal structure to implement Adversorial Neural Network (ANN) to run against reconstruction algorithms of JUNO

## Conventions

This repo expose mutliple classes.

Each classes with the name `I...` is an informal interface, aka. a contract that your class need to implment for them to be usable in the framwork


## Requirements
 - python version 3.9.14

## Documentation

The documentation can be generated using the command
```bash
$ ./doc.sh --build
```

You can start a webserver hosting the doc using
```bash
$ ./doc.sh
Welcome to the JANNE documentation helper
Choose an action to run
 1) Build
 2) Serve
 3) Quit
Choose an action : 2
Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```
