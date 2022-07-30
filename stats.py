from pstats import Stats, SortKey

stats = Stats('stats/0.0 deg.txt')
stats.sort_stats(SortKey.TIME).print_stats()

