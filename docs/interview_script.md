# Fleet Operator Interview Script (15–20 min)

**Goal:** find evidence that could kill the idea. Ask about what they do now and what broke recently, never "would you use our app?"
**Who:** owners or managers of 10–100 vehicles (logistics/delivery, school/staff transport, cabs, field service, small trucking).
**Target:** 10+ interviews by Nov 8. Log every call in `interview_log.csv`.

## Opening (1 min)
"We're students researching how small fleets handle vehicle maintenance. We're not selling anything. Can I ask about how you run things today?"

## 1. Fleet basics
1. How many vehicles do you run? What types, and how old?
2. What do you use for tracking today? (GPS tracker / app / AIS-140 / nothing) Who's the vendor?
3. Roughly what do you pay for it per vehicle per month?

## 2. Maintenance today
4. How do you decide a vehicle needs servicing? (km / calendar / driver complaint / mechanic)
5. Who decides whether a vehicle is taken off the road?
6. Do you get any alerts from your tracker today? Which ones do you ignore, and why?

## 3. Last incident (the most important section)
7. When was your last unexpected breakdown? Walk me through what happened.
8. Were there warning signs before it? Did anyone notice them?
9. What did it cost? (towing, repair, lost trips or deliveries, replacement vehicle, penalties)
10. How many breakdowns like that in the last 3 months?

## 4. Value + price
11. What does one day of a vehicle sitting idle cost you?
12. If something told you each morning which 2–3 vehicles need attention and why, what would you do with that?
13. Would you pay per vehicle per month for that? What range feels fair? (If they stall, ask: ₹199? ₹399?)
14. Would you plug in a small OBD device, or would you only consider something that uses your existing tracker?
15. What would stop you from trying it?

## Close
16. Can we run a free 2–4 week pilot on 2–3 of your vehicles? (Note the exact answer: yes / maybe / no.)
17. Who else runs a fleet that we should talk to?

## After each call: fill one row in the log
`date, name, segment, vehicles, current_tool, tool_cost, last_breakdown, breakdowns_3mo, downtime_cost_per_day, ignores_alerts(y/n), wtp_range, obd_ok(y/n), pilot(yes/maybe/no), quote`

**Red flags to record honestly:** "breakdowns are rare", "my mechanic handles it", "my tracker already does this", "won't pay extra". If most calls say these → pivot (Go/No-Go, section 19 of the strategy doc).
