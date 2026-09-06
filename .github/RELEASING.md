# Registering a release

1. Merge the version bump, release notes and passing tests into `main`.
2. Comment `@JuliaRegistrator register` on the release commit, followed by
   `Release notes:` and the version's notes. Explain numerical or API changes
   and migration steps when the version breaks compatibility.
3. Follow the resulting JuliaRegistries/General pull request until its checks
   pass and the registry merges it.
4. General's `JuliaTagBot` notification triggers `TagBot.yml`, which creates
   the version tag and GitHub release. A manual workflow dispatch can retry it.

TagBot uses the repository-scoped `TAGBOT_SSH_KEY` deploy key to push tags so
the Documentation workflow runs for versioned docs. Its GitHub token uses this
repository's default write workflow permissions. If those defaults change,
check TagBot's permissions before the next release. No personal access token
is stored for this workflow.

The registry records the registered commit's tree. Avoid moving a published
tag or reusing a registered version for changed package contents. If TagBot
reports GitHub's workflow-file release restriction, inspect the actual tag
and registry commit before completing the GitHub release manually.

References: [Registrator](https://github.com/JuliaRegistries/Registrator.jl#how-to-use)
and [TagBot](https://github.com/JuliaRegistries/TagBot#setup).
