# Puck Lab: activity deck copy

Paste-ready text for the "[Activity] Make your own!" template, written for the projected format (hosts operate the tool, teams predict, vote and judge). Numbers marked *measured* were re-run on 30 Sep 2026 and need re-checking on the day.

## 1. Names: main workshop vs the activity

There are two names. The master deck's title slide (the one that currently says "Prompt to Placement") carries the **main workshop title**. The activity template's slide 1 carries the **activity name**.

### Activity name (slide 1 of the activity template): chosen

> **Puck Around, Find Out**
> Try things with AI. Keep what holds up.

It riffs on "f around and find out": the room tries changes, sees what happens, and decides from the evidence. If the pun is too spicy for the venue or the brand, fall back to *Confidently Wrong* with the same subtitle.

### Main workshop title (master deck title slide): chosen

> **Call Your Shot!**
> Use ChatGPT Work to Think, Build, Benchmark

The title has to say that the teams think first and the hosts then see their approach play out, not that the tool does the work. "Call your shot" means saying what you expect before you take it. In the session: the team predicts and commits, the hosts run it, and everyone sees whether it lands. It also carries the puck pun. The subtitle copies the master deck's pattern ("Discover, Tailor & Apply") with three verbs that match the flow: Think (the team's approach), Build (we run the change with ChatGPT Work), Benchmark (the scored evidence). Optional tweaks: write it "Think, Build & Benchmark" to match the master deck exactly, and drop the "!" if you want it to look like the other titles.

Alternatives, if you want a different feel:

| Title | Subtitle | Feel |
|---|---|---|
| Show Your Work | Use ChatGPT Work to Think, Build, Benchmark | The math-class phrase. Teams show their approach and the evidence. |
| Think First, Then Prompt | Use ChatGPT Work to Plan, Build, Benchmark | Plain and explicit: the thinking comes before the tool. |
| Brains Before Bots | Use ChatGPT Work to Think, Build, Benchmark | Cheeky, and it puts the team's thinking ahead of the tool. |

Pick one title and use the same words on the check-in slide, the title slide and any poster.

Use the same product name ("ChatGPT Work", as in the deck) everywhere: slides, poster, briefing and runbook.

## 2. Slides 2 to 4: delete

These are instructions to the host ("[DELETE] Design your activity", "Build and test it", "Show the team"). Remove them before the session. One item is worth keeping from slide 4: share the deck with the team before running it.

## 3. Slide 5: "What did you make, and why was it useful?"

The template gives you a photo, a name, and one big box. The left side only fits a short line.

- **Under "Your name" (about 25 words):** *I used ChatGPT Work to find out why a puck model kept missing, tried a fix, and measured it: 2/24 targets became 13/24.*
- **Big box:** a Lab screenshot after a missed shot (dashed violet prediction, solid blue path, dotted line between the endpoints). Caption: *Same push. The model predicted it would land at the edge of the ring; reality disagreed.*
- *Measured 30 Sep 2026 on development scenes. The mean miss also fell from 0.385 to 0.101 table widths. One model, one simulator, one seed: do not present it as a general result.*

## 4. Slide 6: "What could students try?"

| Use case | One line | Image to add |
|---|---|---|
| **Diagnose** | Find where the model's prediction and reality split apart. | Lab screenshot of a missed shot with the three result cards |
| **Repair** | Try one change to how the network is trained, and compare. | Terminal output of `run_experiment.py` showing the score and "new successes / new failures" |
| **Stress-test** | Move the target, cap the push speed, add more search. Does it still work? | Experiment log table from the Lab with a few shots |

## 5. Slide 7: "Choose your first idea"

Three questions for the teams (they discuss, then vote):

1. What do you think the model gets wrong first: collisions with the bumper, long stretches of free motion, or the shots the planner picks?
2. What would you change: how much collisions count, what the network can see, or how it moves when nothing is touching?
3. What result would prove you wrong?

## 6. Slide 8: starter prompt

The slide fits about 60 words. The host pastes this into ChatGPT Work. It is read-only, so nothing can break.

> Read student/lab.py without editing. Find the function that picks the push and the one that predicts paths. Explain simply what each does and which one uses the neural network. Then run the original network on the Across target (start 0.16, 0.70; target 0.83, 0.72) and show the predicted miss next to the actual miss. Don't edit any files. Say what you didn't check.

Expected for the reference shot (*measured*): predicted miss 6.8%, actual miss 29.1%.

## 7. Slide 9: "Discuss with your group"

Adapted from the template's three questions so they match the session:

1. What did the model **predict**, and what **actually** happened?
2. **Where** do the two paths differ, and why might that be?
3. What **one change** would you try, and what result would change your mind?

(The template's original questions also fit if you prefer them: What already works? What feels generic or incomplete? What is one improvement you want to make?)

## 8. Slide 14: "Take your idea further"

The three columns hold about 30 words each, so the slide uses short versions. Improvements 1 and 2 are worked examples the hosts run. **Improvement 3 is the teams' own prompt** (you can rename its heading "Your idea"); the 96-scene real test moves to slide 15 and is run by the hosts.

| | **Slide text (short)** | **Full prompt to paste into the tool** |
|---|---|---|
| **Improvement 1** | Run the baseline, then the same model with collision loss weighted 5x. Keep the planner, scenes and seed fixed. Show both scores, mean miss, and new successes and failures. | From student/, run `python run_experiment.py --name baseline`, then `python run_experiment.py --name contacts --contact-weight 5 --compare baseline`. Do not change the planner, scenes, seed or evaluator. Show both scores, the mean miss, and every new success and new failure. |
| **Improvement 2** | Now run the free-motion head against the baseline. Explain in plain words why it might cut error that builds up over 35 steps. Don't claim a cause. | Run `python run_experiment.py --name free-motion --free-head --compare baseline`. Show what code path it adds, explain in plain words why it might reduce error that builds up over 35 steps, and do not claim a cause. |
| **Improvement 3 (Your idea)** | **Your idea:** write your own follow-up prompt. Change ONE thing. Say what you expect and what would change your mind. Hand it to us and we'll run it live. | The team's own prompt, plus the guardrail suffix from the runbook (section 6.5b). |

Optional hint line under the columns: *Stuck? Try a wider network, more training passes, or extra inputs about the walls and the bumper. Or ask GPT for ideas.*

**Real test (slide 15, run by the hosts in the full `puck-lab` folder):** "Freeze the candidate. Score the baseline and the candidate once each on the 96 held-out scenes with `python host/score_submission.py`. Do not tune after seeing the result. Report hits, mean miss, and paired new successes and regressions, and say what this does not show." For reference (*measured*): the baseline and the free-motion repair scored 6/96 and 69/96 on that set. Do not put these on a slide before the teams see them.

## 8b. Optional new slide after slide 14: "Your turn"

The plan now has teams write their own prompt. If you add a slide for it, use this:

> **Your turn: write the prompt**
> Pick ONE change. Say what you expect, and what would change your mind.
> - **Goal:** what you want to improve
> - **Change:** one thing
> - **Keep fixed:** the planner, the scenes, the seed
> - **Check:** hits, mean miss, new successes and new failures vs the baseline

Speaker line: "Now it's your turn. Use your own brains. Four minutes. Spokespeople, we'll read your prompts and run two of them live."

## 9. Changes to the other slides in the master deck

| Slide in the master deck | Change |
|---|---|
| Form a group ("groups of 2 to 3") | Our format uses **10 teams of about 7** (one vote per team, roles: lead, recorder, spokesperson). Change the line, or you will get 25 or more groups to count votes from. |
| ChatGPT Work | Keep. Optionally add one line: "Today the hosts run it; your team decides what to try." |
| Activity (slide 8) | Replace with: *You'll direct ChatGPT Work to investigate a puck-planning model. Your team predicts, votes on what to try, and we run it live. Then you compare the results and decide: did the change actually help?* |
| Check out, stay connected | Keep as they are. |
| Resume example slides | Delete. They belong to the other workshop. |

## 10. The pieces to take from the other deck

- The "starter prompt, then improvements" shape works. Ours has the same structure.
- The master deck's example prompts name constraints ("do not invent skills or metrics"). Ours do the same ("do not edit files", "do not tune on the test set"). Keep that habit; it is the lesson.
