# McNerd-ML
Some useful code for working with machine learning techniques in C#

## Background
In the summer of 2016, I completed Andrew Ng's excellent introduction to Machine Learning course on Coursera.
It was also my first introduction to Octave/MATLAB, which made light work of many of the calculations involved.

After the course, I looked around for C# libraries so I could apply the techniques on a platform that was more
familiar to me. There are some pretty good ones out there! Even so, I thought it would be a great exercise to
try and implement my own. The intention is not to compete with any of the established libraries, but to practice
my own coding skills, and probably blog about it now and then.

As a result, this library is focussed primarily on handling the assignments set in the above course, and I'll
most likely apply them to some Kaggle datasets too. Hopefully in time it will become more advanced, but for now
I'm not going to think that far ahead lest it become too overwhelming!

## Usage

The current version of this library was created using Visual Studio Community 2015. Download the code and open
the solution (.sln) file. It contains three projects:

* The class library itself.
* A console application, just used for quick testing and profiling.
* Unit Tests using the built in unit testing framework.

The class library is the only part required if you're interested in adding this to your own applications.
Build it and add a reference to it. Done.

## Performance

I'm running the CLI version of Octave on the same PC I'm using to develop this library. Every time I write
something new, I will try approximate it using Octave's built-in functions for comparison. If a method takes
significantly longer in my code, I'll look at various methods to bring it approximately in line with Octave.
However, I tend to get a bit obsessive, so once improvements start happening, I'll probably keep going until
it outperforms Octave.

Note that some performance boosts include parallel processing, so your specific mileage may vary.

## Contact Me

I can be contacted through [my webpage](http://mrmcnerd.com/contact/), [Twitter](https://twitter.com/nerdbehaviour),
and [StackOverflow](http://stackoverflow.com/users/5233918/michael-mcmullin).
