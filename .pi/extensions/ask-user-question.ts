/**
 * Claude Code-style AskUserQuestion tool for pi.
 *
 * Supports one or more questions, option descriptions, single-select and
 * multi-select answers, and a final review/submit screen for multi-question
 * prompts. The tool schema intentionally mirrors Claude Code's
 * AskUserQuestion input shape:
 *
 * {
 *   questions: [{
 *     header?: string,
 *     question: string,
 *     multiSelect?: boolean,
 *     options: [{ label: string, description?: string }]
 *   }]
 * }
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Text, Key, matchesKey, truncateToWidth } from "@mariozechner/pi-tui";
import { Type } from "typebox";

interface QuestionOption {
  label: string;
  description?: string;
}

interface UserQuestion {
  question: string;
  header?: string;
  multiSelect?: boolean;
  options: QuestionOption[];
}

interface QuestionAnswer {
  question: string;
  header?: string;
  multiSelect: boolean;
  selected: string[];
  selectedIndexes: number[];
}

interface AskUserQuestionDetails {
  questions: UserQuestion[];
  answers: Record<string, string | string[]>;
  selections: QuestionAnswer[];
  cancelled: boolean;
}

const OptionSchema = Type.Object({
  label: Type.String({ description: "Display label for this option" }),
  description: Type.Optional(
    Type.String({
      description: "Optional longer explanation shown below the label",
    }),
  ),
});

const QuestionSchema = Type.Object({
  header: Type.Optional(
    Type.String({
      description: "Short title/context for the question, shown in the UI",
    }),
  ),
  question: Type.String({
    description: "The concrete question the user must answer",
  }),
  multiSelect: Type.Optional(
    Type.Boolean({
      description:
        "Whether the user may choose multiple options. Defaults to false.",
    }),
  ),
  options: Type.Array(OptionSchema, {
    minItems: 1,
    description:
      "Answer options. Include a recommended/default option when appropriate.",
  }),
});

const AskUserQuestionParams = Type.Object({
  questions: Type.Array(QuestionSchema, {
    minItems: 1,
    description: "One or more questions to ask the user before proceeding",
  }),
});

function wrapPlain(text: string, width: number): string[] {
  const safeWidth = Math.max(10, width);
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return [""];

  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    if (word.length > safeWidth) {
      if (current) {
        lines.push(current);
        current = "";
      }
      for (let i = 0; i < word.length; i += safeWidth) {
        lines.push(word.slice(i, i + safeWidth));
      }
      continue;
    }
    const next = current ? `${current} ${word}` : word;
    if (next.length > safeWidth) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function makeDetails(
  questions: UserQuestion[],
  selectedByQuestion: Array<Set<number>>,
  cancelled: boolean,
): AskUserQuestionDetails {
  const selections: QuestionAnswer[] = questions.map(
    (question, questionIndex) => {
      const selectedIndexes = Array.from(
        selectedByQuestion[questionIndex] ?? [],
      )
        .sort((a, b) => a - b)
        .map((index) => index + 1);
      const selected = selectedIndexes.map(
        (index) => question.options[index - 1]?.label ?? `Option ${index}`,
      );
      return {
        question: question.question,
        header: question.header,
        multiSelect: question.multiSelect === true,
        selected,
        selectedIndexes,
      };
    },
  );

  const answers: Record<string, string | string[]> = {};
  for (const selection of selections) {
    answers[selection.question] = selection.multiSelect
      ? selection.selected
      : (selection.selected[0] ?? "");
  }

  return { questions, answers, selections, cancelled };
}

function formatAnswerSummary(details: AskUserQuestionDetails): string {
  if (details.cancelled) return "User cancelled the question dialog.";

  const parts = details.selections.map((selection) => {
    const question = JSON.stringify(selection.question);
    const answer = selection.multiSelect
      ? JSON.stringify(selection.selected)
      : JSON.stringify(selection.selected[0] ?? "");
    return `${question}=${answer}`;
  });

  return `User has answered your questions: ${parts.join(", ")}. You can now continue with the user's answers in mind.`;
}

export default function askUserQuestion(pi: ExtensionAPI) {
  pi.registerTool({
    name: "AskUserQuestion",
    label: "Ask User Question",
    description:
      "Ask the user one or more structured clarification questions with selectable options. Use this instead of inline prose questions when you need a decision before continuing.",
    promptSnippet:
      "Ask the user structured clarification questions with selectable options",
    promptGuidelines: [
      "Use AskUserQuestion for concrete user decision points, ambiguity, or choices instead of asking inline prose questions.",
      "AskUserQuestion questions should be specific, concise, and actionable; include enough context for the user to decide.",
      "AskUserQuestion options should be mutually distinct; mark a recommended/default option in the label when you have one.",
      "Use AskUserQuestion multiSelect=true only when multiple options can validly be chosen together.",
    ],
    parameters: AskUserQuestionParams,

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const questions = params.questions.map((question) => ({
        ...question,
        multiSelect: question.multiSelect === true,
      })) as UserQuestion[];

      if (!ctx.hasUI) {
        throw new Error("AskUserQuestion requires interactive UI mode.");
      }

      if (
        questions.length === 0 ||
        questions.some((question) => question.options.length === 0)
      ) {
        throw new Error(
          "AskUserQuestion needs at least one option for every question.",
        );
      }

      const selectedByQuestion = questions.map(() => new Set<number>());
      const submitTabIndex = questions.length;
      const result = (await ctx.ui.custom((tui, theme, _keybindings, done) => {
        let currentQuestionIndex = 0;
        let optionIndex = 0;
        let cachedLines: string[] | undefined;

        function invalidate() {
          cachedLines = undefined;
        }

        function refresh() {
          invalidate();
          tui.requestRender();
        }

        function isSubmitTab() {
          return currentQuestionIndex === submitTabIndex;
        }

        function currentQuestion() {
          return questions[
            Math.min(currentQuestionIndex, questions.length - 1)
          ];
        }

        function currentSelections() {
          return selectedByQuestion[
            Math.min(currentQuestionIndex, questions.length - 1)
          ];
        }

        function answered(index: number) {
          return (selectedByQuestion[index]?.size ?? 0) > 0;
        }

        function allAnswered() {
          return questions.every((_question, index) => answered(index));
        }

        function firstUnansweredIndex() {
          return questions.findIndex((_question, index) => !answered(index));
        }

        function clampOptionIndex() {
          if (isSubmitTab()) {
            optionIndex = 0;
            return;
          }
          optionIndex = Math.max(
            0,
            Math.min(optionIndex, currentQuestion().options.length - 1),
          );
        }

        function goToQuestion(index: number) {
          currentQuestionIndex = Math.max(0, Math.min(index, submitTabIndex));
          optionIndex = 0;
          clampOptionIndex();
          refresh();
        }

        function advance() {
          const missing = firstUnansweredIndex();
          if (missing >= 0 && missing > currentQuestionIndex) {
            goToQuestion(missing);
            return;
          }
          if (questions.length === 1 && allAnswered()) {
            done(makeDetails(questions, selectedByQuestion, false));
            return;
          }
          if (currentQuestionIndex < questions.length - 1) {
            goToQuestion(currentQuestionIndex + 1);
            return;
          }
          goToQuestion(submitTabIndex);
        }

        function toggleCurrentOption() {
          const question = currentQuestion();
          const selections = currentSelections();
          if (question.multiSelect) {
            if (selections.has(optionIndex)) selections.delete(optionIndex);
            else selections.add(optionIndex);
          } else {
            selections.clear();
            selections.add(optionIndex);
          }
        }

        function handleInput(data: string) {
          if (matchesKey(data, Key.escape)) {
            done(makeDetails(questions, selectedByQuestion, true));
            return;
          }

          if (matchesKey(data, Key.tab) || matchesKey(data, Key.right)) {
            goToQuestion((currentQuestionIndex + 1) % (questions.length + 1));
            return;
          }
          if (
            matchesKey(data, Key.shift("tab")) ||
            matchesKey(data, Key.left)
          ) {
            goToQuestion(
              (currentQuestionIndex - 1 + questions.length + 1) %
                (questions.length + 1),
            );
            return;
          }

          if (isSubmitTab()) {
            if (matchesKey(data, Key.enter)) {
              if (allAnswered())
                done(makeDetails(questions, selectedByQuestion, false));
              else goToQuestion(Math.max(0, firstUnansweredIndex()));
            }
            return;
          }

          const question = currentQuestion();
          if (matchesKey(data, Key.up)) {
            optionIndex = Math.max(0, optionIndex - 1);
            refresh();
            return;
          }
          if (matchesKey(data, Key.down)) {
            optionIndex = Math.min(
              question.options.length - 1,
              optionIndex + 1,
            );
            refresh();
            return;
          }
          if (matchesKey(data, Key.space)) {
            toggleCurrentOption();
            refresh();
            return;
          }
          if (matchesKey(data, Key.enter)) {
            if (question.multiSelect) {
              if (!answered(currentQuestionIndex)) {
                toggleCurrentOption();
                refresh();
                return;
              }
              advance();
              return;
            }
            toggleCurrentOption();
            advance();
          }
        }

        function addWrapped(
          lines: string[],
          text: string,
          width: number,
          indent: string,
          color: (s: string) => string,
        ) {
          for (const wrapped of wrapPlain(
            text,
            Math.max(10, width - indent.length),
          )) {
            lines.push(truncateToWidth(indent + color(wrapped), width));
          }
        }

        function renderTabs(lines: string[], width: number) {
          if (questions.length <= 1) return;
          const parts = questions.map((question, index) => {
            const active = index === currentQuestionIndex;
            const mark = answered(index) ? "✓" : "□";
            const label = question.header || `Q${index + 1}`;
            const text = ` ${mark} ${label} `;
            if (active) return theme.bg("selectedBg", theme.fg("text", text));
            return theme.fg(answered(index) ? "success" : "muted", text);
          });
          const submitText = " Submit ";
          parts.push(
            isSubmitTab()
              ? theme.bg("selectedBg", theme.fg("text", submitText))
              : theme.fg(allAnswered() ? "success" : "dim", submitText),
          );
          lines.push(truncateToWidth(` ${parts.join(" ")}`, width));
          lines.push("");
        }

        function renderQuestion(lines: string[], width: number) {
          const question = currentQuestion();
          const title =
            question.header || `Question ${currentQuestionIndex + 1}`;
          const mode = question.multiSelect ? "multi-select" : "single-select";
          lines.push(
            truncateToWidth(
              theme.fg("accent", theme.bold(` ${title}`)) +
                theme.fg("dim", ` (${mode})`),
              width,
            ),
          );
          addWrapped(lines, question.question, width, " ", (s) =>
            theme.fg("text", s),
          );
          lines.push("");

          const selections = currentSelections();
          for (let index = 0; index < question.options.length; index++) {
            const option = question.options[index];
            const cursor = index === optionIndex ? "> " : "  ";
            const checked = selections.has(index);
            const marker = question.multiSelect
              ? checked
                ? "[x]"
                : "[ ]"
              : checked
                ? "(●)"
                : "( )";
            const color =
              index === optionIndex ? "accent" : checked ? "success" : "text";
            addWrapped(
              lines,
              `${marker} ${index + 1}. ${option.label}`,
              width,
              cursor,
              (s) => theme.fg(color, s),
            );
            if (option.description) {
              addWrapped(lines, option.description, width, "      ", (s) =>
                theme.fg("muted", s),
              );
            }
          }
          lines.push("");
          const help = question.multiSelect
            ? "↑↓ navigate • Space toggle • Enter next/submit • Tab switch • Esc cancel"
            : "↑↓ navigate • Enter select • Tab switch • Esc cancel";
          lines.push(truncateToWidth(theme.fg("dim", ` ${help}`), width));
        }

        function renderSubmit(lines: string[], width: number) {
          lines.push(
            truncateToWidth(
              theme.fg("accent", theme.bold(" Review answers")),
              width,
            ),
          );
          lines.push("");
          for (let index = 0; index < questions.length; index++) {
            const question = questions[index];
            const labels = Array.from(selectedByQuestion[index] ?? [])
              .sort((a, b) => a - b)
              .map(
                (selectedIndex) =>
                  question.options[selectedIndex]?.label ??
                  `Option ${selectedIndex + 1}`,
              );
            const prefix = answered(index)
              ? theme.fg("success", "✓")
              : theme.fg("warning", "!");
            const title = question.header || `Question ${index + 1}`;
            addWrapped(
              lines,
              `${title}: ${labels.join(", ") || "unanswered"}`,
              width,
              ` ${prefix} `,
              (s) => theme.fg(answered(index) ? "text" : "warning", s),
            );
          }
          lines.push("");
          if (allAnswered()) {
            lines.push(
              truncateToWidth(
                theme.fg(
                  "success",
                  " Enter to submit • Tab/←→ revise • Esc cancel",
                ),
                width,
              ),
            );
          } else {
            lines.push(
              truncateToWidth(
                theme.fg(
                  "warning",
                  " Enter jumps to first unanswered • Tab/←→ revise • Esc cancel",
                ),
                width,
              ),
            );
          }
        }

        function render(width: number) {
          if (cachedLines) return cachedLines;
          const safeWidth = Math.max(1, width);
          const lines: string[] = [];
          lines.push(theme.fg("accent", "─".repeat(safeWidth)));
          renderTabs(lines, safeWidth);
          if (isSubmitTab()) renderSubmit(lines, safeWidth);
          else renderQuestion(lines, safeWidth);
          lines.push(theme.fg("accent", "─".repeat(safeWidth)));
          cachedLines = lines.map((line) => truncateToWidth(line, safeWidth));
          return cachedLines;
        }

        return { render, handleInput, invalidate };
      })) as AskUserQuestionDetails | undefined;

      if (!result) {
        throw new Error(
          "AskUserQuestion custom UI is unavailable in this mode.",
        );
      }

      if (result.cancelled) {
        return {
          content: [{ type: "text", text: formatAnswerSummary(result) }],
          details: result,
        };
      }

      return {
        content: [{ type: "text", text: formatAnswerSummary(result) }],
        details: result,
      };
    },

    renderCall(args, theme, _context) {
      const questions = Array.isArray(args.questions)
        ? (args.questions as UserQuestion[])
        : [];
      const count = questions.length;
      const labels = questions
        .map((question, index) => question.header || `Q${index + 1}`)
        .join(", ");
      let text = theme.fg("toolTitle", theme.bold("AskUserQuestion "));
      text += theme.fg("muted", `${count} question${count === 1 ? "" : "s"}`);
      if (labels) text += theme.fg("dim", ` (${truncateToWidth(labels, 50)})`);
      return new Text(text, 0, 0);
    },

    renderResult(result, _options, theme, _context) {
      const details = result.details as AskUserQuestionDetails | undefined;
      if (!details) {
        const text = result.content[0];
        return new Text(text?.type === "text" ? text.text : "", 0, 0);
      }
      if (details.cancelled)
        return new Text(theme.fg("warning", "Cancelled"), 0, 0);

      const lines = details.selections.map((selection) => {
        const label = selection.header || selection.question;
        const answer = selection.multiSelect
          ? selection.selected.join(", ")
          : (selection.selected[0] ?? "");
        return `${theme.fg("success", "✓ ")}${theme.fg("accent", label)}: ${theme.fg("text", answer)}`;
      });
      return new Text(lines.join("\n"), 0, 0);
    },
  } as any);
}
