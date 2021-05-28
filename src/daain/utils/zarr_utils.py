def open_group(group, path):
    """Creates or loads the subgroup defined by `path`."""
    if path in group:
        return group[path]
    else:
        return group.create_group(path)
