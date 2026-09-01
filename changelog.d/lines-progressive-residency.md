#### Additive Lines ladders release decoded rung buffers after concatenation

The progressive Lines loader now retains one cumulative geometry payload instead
of keeping every decoded ladder rung beside the concatenated result. A fully
loaded additive ladder therefore returns to approximately the same terminal
geometry residency as an unladdered Lines leaf rather than holding both copies.
Logical ladder depth is tracked separately so refinement, quality stamps,
monitoring, and slice-cache restores continue from the correct next rung even
when several rungs have been folded into one cached payload.
