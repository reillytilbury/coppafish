## The Algorithm

Coppafish is built on the basis that an algorithm that performs well does not need to be changed. The algorithm is only 
updated when there is evidence that it can perform better and that the current algorithm is performing worse.

## Our Philosophy

We follow some basic rules when coding. Anyone can code something that works, but coding it in a scaleable, 
maintainable way is another struggle altogether. Knowledge written down in code twice is bad code. Don't Repeat 
Yourself (DRY)!

Here are some specific standards to follow:

* Every time a function is modified or created, a new unit test must be created for the function. A pre-existing unit 
test can be drawn from to build a new unit test, but it should be clear in your mind that you are affectively building 
a new function.
* Minimise `#!python if`/`#!python else` branching as much as possible. Exit `#!python if`/`#!python else` nesting as 
soon as possible through the use of keywords like `continue`, `break` and `return` whenever feasible.
* Every docstring for a function must be complete so a developer can re-create the function without seeing any of the 
existing source code.
* Each parameter in a function must have an independent, clear functionality. If two parameters are derivable from 
one another, you are doing something wrong.
* Minimise the number of data types a parameter can be and use common sense. For example, a parameter that can be 
`#!python int` or `#!python None` is reasonable. A parameter that can be `#!python bool` or `#!python float` is not 
reasonable.
* The documentation should update in parallel with the code. Having the documentation as part of the github repository 
is done to make this as easy as possible. 
* Do not over-shorten a variable or function name.
* In most cases, a line of code should do only one operation.
