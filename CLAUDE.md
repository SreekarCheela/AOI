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
