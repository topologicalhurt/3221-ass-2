# 3221-ass-2
Implement a simple Federated Learning (FL) system including five clients in total and one server for aggregation 

# Instructions

There are 2 choices - in production mode the intended way of launching the program is using main.py (_See I_) otherwise
invoke Comp3221_FLServer.py directly in debug mode

### Method I

Invoke the main.py script to run the server in a predictable way I.E runs Comp3221_FLServer.py with the following args:
* -O (launches the server in non-debug mode)
* -u (unbuffered stdout & stderr for friendly multiple server instances)
* Fixed port of 6000 (when -O is specified the port number is a positional argument)

Invoke main as follows: **main.py [-h] sub_client**

### Method II

When invoked without the -O argument (in debug mode) the program is run like:

**Comp3221_FLServer.py [-h] [--port_no PORT_NO] sub_client**

And when the program is invoked with the -O argument (in production mode) the program is run like:

**Comp3221_FLServer.py [-h] port_no sub_client**

_Note: the only intended use-case for running without the -O argument is adhering to the program specification. 
You probably want to use main.py!!!_





