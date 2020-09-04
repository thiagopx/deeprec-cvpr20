def flatten(l, level=1):
   sublist = l
   for _ in xrange(level):
      sublist = [item for sublist2 in sublist for item in sublist2]
   return sublist