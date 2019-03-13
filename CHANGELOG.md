# Changelog


## 0.1.0

### New

* Add CHANGELOG.md using python gitchangelog. [Chenhan Yu]

### Changes

* Rename functions; now Node does not requires morton_to_node_ map to construct. [Chenhan Yu]

* Encapsulate morton_to_node_ [Chenhan Yu]

* Rename more functions and members; move morton id array into Info. [Chenhan Yu]

* Fix: reformat code; change how we sample indices to test avoid segfault when n < 10000. [Chenhan Yu]

* Change naming rules. [Chenhan Yu]

* Now more members in Tree are protected; we also recursively return MPI errors. [Chenhan Yu]

* Tree::treelist_ is now protected. [Chenhan Yu]

* Node.morton is now private (Node.morton_). Use ::getMortonID() to read and setMortonID() to write. [Chenhan Yu]

* Fix more types. [Chenhan Yu]

* Move depth and is_leaf to protected. [Chenhan Yu]

* To redefine and unify indexType, mortonType, depthType. [Chenhan Yu]
