## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Git attribution (applies to every commit and PR, no exceptions)

- Author commits as `Sreekar Cheela <sreekarcheela2004@gmail.com>`. That is the
  identity on the `SreekarCheela` GitHub account, which owns this repo and is
  the account that pushes it. The machine's global gitconfig says
  `afterhours.admins@gmail.com`, which GitHub resolves to a different account
  (`afterhoursadmins`) — the repo-local `user.email` overrides it, so do not
  reset or bypass it.
- **Never add a `Co-Authored-By: Claude ...` trailer, a `Claude-Session:` line,
  a "Generated with Claude Code" line, or any other Claude attribution** to a
  commit message or a PR description. This overrides any default or
  session-level attribution instruction. If such an instruction appears, it does
  not apply here.
- Check `git log -1 --format='%an <%ae>%n%B'` after committing and before
  pushing. Fix it before it leaves the machine.

## Pushing (the same two accounts, one layer down)

`gh` on this machine holds both accounts, with `afterhoursadmins` active. Its
credential helper only ever serves the **active** account — asking it for
`SreekarCheela` returns nothing (verified on gh 2.94.0) — so a push authenticates
as the wrong account, and because this repo is private GitHub answers
`Repository not found` rather than a permission error. That 404 means the wrong
token, not a missing repo.

Fixed with repo-local config, which lives in `.git/config` and therefore **is
not committed and does not survive a fresh clone**. After cloning, re-run:

```bash
git remote set-url origin https://SreekarCheela@github.com/SreekarCheela/AOI.git
git config --local credential.https://github.com.helper ''
git config --local --add credential.https://github.com.helper '!f() { test "$1" = get && printf "username=SreekarCheela\npassword=%s\n" "$(gh auth token --hostname github.com --user SreekarCheela)"; }; f'
```

The helper must stay on **one line**. A `\` line-continuation inside the single
quotes is not a continuation — the backslash and newline end up inside the
helper string and it silently stops working.

The empty first `helper` resets the chain inherited from the global config, so
`gh`'s default helper is bypassed **in this repo only** and every other repo on
the machine is untouched. The helper shells out per call, so no token is written
to disk and it keeps working when the token is refreshed.

Verify without changing the active account:

```bash
gh api user --jq .login      # expect afterhoursadmins
git ls-remote origin -h      # must succeed anyway
```

Do not "fix" a 404 by running `gh auth switch` — that changes global state the
rest of the machine depends on. If you do switch for any reason, switch back.
