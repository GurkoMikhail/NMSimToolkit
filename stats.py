from pstats import Stats, SortKey

stats = Stats('stats.txt')
stats.sort_stats(SortKey.TIME).print_stats()

