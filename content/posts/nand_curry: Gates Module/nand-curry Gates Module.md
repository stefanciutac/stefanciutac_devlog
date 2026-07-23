---

title: "nand-curry: Gates Module"
date: 2026-07-23
draft: "false"
math: "true"

---

# Introduction

The `Gates` module, predictably, contains the definition of the base `nandGate` function on which all other definitions rely, as well as the definitions of the other logical gates—all defined as wrappers for compositions of the `nandGate` function, or as wrappers for wrappers.

# Design Decisions

Even (or perhaps, "especially for") for a module as simple as this, there are several key decisions that had to be made regarding implementation; they might appear trivial or inconsequential—and perhaps they are—but choosing poorly at this stage could lead to annoyances and regret later in the project.

## Bit Representation

Should bits be Boolean or integer values? It would most likely be marginally quicker to use Integers values should I find myself typing out bit strings for testing purposes; however, if we are being pedantic (which we are), logic gates are Boolean by definition: the inputs are permutations of the Boolean set, and the output is always a Boolean value. This does not actually mean anything, because we can simply define $\{0, 1\}$ to be the Boolean set, and it is equivalent to using $\{False, True\}$.

Ultimately, I decided to use the latter of these two sets, because it means the type signatures of all of the functions I define are completely correct;

```haskell
nandGate :: Bool -> Bool -> Bool
```

is more accurate than:

```haskell
nandGate :: Int -> Int -> Int
```

since the latter implies that `nandGate` can take any value of the `Int` type and return any value of the `Int` type. Furthermore, it suggests that the output of the function is of the `Num` class, and so can have numerical operators applied to it: it would not make any sense.

The best solution would probably be to define a custom type which can only take integer or character values in the set $\{0, 1\}$, but I find that it is generally better to avoid adding complication if the benefit is largely aesthetic or to avoid some inconcrete, potential exertion: I simply chose to use the built-in `Bool` type for the sake of simplicity and correctness.

## Derived Logic Gate Definition

Should the other logic gates that are not the NAND gate be defined entirely with the `nandGate` function, or should wrappers (other gates) be used where possible? This decision is largely inconsequential, but establishing a convention (whether to always use functions from lower layers of abstraction or to mix layers of abstraction for conciseness where possible) in this module would inform the definition "style guide" for more complex functions.

It does not make sense for the entire project to be based around defining incremental layers of abstraction using wrappers defined (if you go back far enough) in terms of the `nandGate` function, only to shun abstraction in some capacity when it comes to function definitions within a particular module. One must choose an extreme here: either everything is defined solely using the original `nandGate` function, or it is deemed perfectly fine for wrappers to be used everywhere. And since wrappers are (by definition) logically equivalent to their definitions, convenience dictates that the most concise syntax can and should be used everywhere, with no repercussions to syntax readability or functionality.

For example, I chose to define the OR gate like this:

```haskell
orGate a b = nandGate (notGate a) (notGate b)
```

rather than like this:

```haskell
orGate a b = nandGate (nandGate a a) (nandGate b b)
```

## Increased-Arity AND Gates

I chose to define separate variations of the AND gate (`andGate`, `and3Gate`, and `and4Gate`—with arity 2, 3, and 4, respectively) rather than composing `andGate` in the modules where I needed AND operators for 3 or 4 inputs. This follows directly from the convention established above, of using wrappers wherever they make syntax easier to read and write.

# Type Signatures

Below are the type signatures of all the functions in the `Gates` module.

```haskell
nandGate :: Bool -> Bool -> Bool
andGate :: Bool -> Bool -> Bool
and3Gate :: Bool -> Bool -> Bool -> Bool
and4Gate :: Bool -> Bool -> Bool -> Bool -> Bool
notGate :: Bool -> Bool
orGate :: Bool -> Bool -> Bool
xorGate :: Bool -> Bool -> Bool
```

The implementation of these functions can be found in the `Gates.hs` module of the `nand-gate` repository on the GitHub linked from the landing page of this website.