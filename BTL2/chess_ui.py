
import tkinter as tk
import chess
import time
import threading
import math
import random
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
from rl_agent import RLAgent
from rl_agent import RLAgent
# Optional sound support on Windows
try:
	import winsound
except Exception:
	winsound = None


# Unicode pieces mapping
UNICODE_PIECES = {
	'K': '\u2654', 'Q': '\u2655', 'R': '\u2656', 'B': '\u2657', 'N': '\u2658', 'P': '\u2659',
	'k': '\u265A', 'q': '\u265B', 'r': '\u265C', 'b': '\u265D', 'n': '\u265E', 'p': '\u265F'
}


class ChessUI:
	def __init__(self, root, white_agent=None, black_agent=None, square_size=64, delay=400):
		self.root = root
		self.square_size = square_size
		self.delay = delay  # milliseconds between moves when running

		self.board = chess.Board()
		self.white_agent = white_agent or RandomAgent()
		self.black_agent = black_agent or MinimaxAgent()

		width = height = self.square_size * 8 + 40
		self.canvas = tk.Canvas(root, width=width, height=height)
		self.canvas.pack()

		self.status_var = tk.StringVar()
		self.status = tk.Label(root, textvariable=self.status_var)
		self.status.pack()

		controls = tk.Frame(root)
		controls.pack()
		self.start_btn = tk.Button(controls, text="Start", command=self.start_game)
		self.start_btn.pack(side=tk.LEFT)
		self.step_btn = tk.Button(controls, text="Step", command=self.step, state=tk.NORMAL)
		self.step_btn.pack(side=tk.LEFT)
		self.reset_btn = tk.Button(controls, text="Reset", command=self.reset)
		self.reset_btn.pack(side=tk.LEFT)

		self.is_running = False

		self.draw_board()
		self.draw_pieces()

	def draw_board(self):
		self.canvas.delete('square')
		# colors tuned to match the example: light cream and olive green
		# slightly darker light color so white pieces show up better
		light = '#E6E2C8'
		dark = '#6F9B4B'
		margin = 20
		s = self.square_size
		# draw outer border
		board_x1 = margin - 6
		board_y1 = margin - 6
		board_x2 = margin + 8*s + 6
		board_y2 = margin + 8*s + 6
		self.canvas.create_rectangle(board_x1, board_y1, board_x2, board_y2, fill='#2f2f2f', outline='')
		# draw squares; ranks 8..1 from top to bottom
		for rank in range(8):
			for file in range(8):
				x1 = margin + file * s
				y1 = margin + rank * s
				x2 = x1 + s
				y2 = y1 + s
				color = light if (file + rank) % 2 == 0 else dark
				self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, width=0, tags='square')
				# subtle inner border to match style
				self.canvas.create_line(x1, y1, x2, y1, fill='#3b3b3b', width=0.7)
				self.canvas.create_line(x1, y1, x1, y2, fill='#3b3b3b', width=0.7)

		# file labels inside small rounded boxes at bottom
		files = 'abcdefgh'
		label_h = 18
		for i, f in enumerate(files):
			x1 = margin + i * s + s*0.08
			x2 = x1 + s*0.84
			y1 = margin + 8*s + 6
			y2 = y1 + label_h
			self.canvas.create_rectangle(x1, y1, x2, y2, fill=light, outline='#2f2f2f', tags='square')
			self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=f, font=('Helvetica', 10, 'bold'))
		# rank labels inside small boxes on the left
		for r in range(8):
			x1 = 6
			x2 = 6 + 18
			y1 = margin + r * s + (s - label_h)/2
			y2 = y1 + label_h
			self.canvas.create_rectangle(x1, y1, x2, y2, fill=light, outline='#2f2f2f', tags='square')
			self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=str(8 - r), font=('Helvetica', 10, 'bold'))

	def draw_pieces(self):
		# kept for backward compatibility — implementation with exclude is below
		return

	def draw_pieces(self, exclude_square=None):
		"""Draw pieces but optionally exclude a square (useful for animation)."""
		self.canvas.delete('piece')
		margin = 20
		s = self.square_size
		for rank in range(8):
			for file in range(8):
				sq = chess.square(file, 7 - rank)
				if exclude_square is not None and sq == exclude_square:
					continue
				piece = self.board.piece_at(sq)
				if piece:
					symbol = piece.symbol()
					glyph = UNICODE_PIECES.get(symbol, '?')
					x = margin + file * s + s/2
					y = margin + rank * s + s/2
					# draw a small shadow/outline to make both white and black pieces visible
					if symbol.isupper():
						# White piece: draw two black shadow layers then white glyph for stronger contrast
						self.canvas.create_text(x+2, y+2, text=glyph, font=('Segoe UI Symbol', int(s*0.72)), fill='black', tags='piece')
						self.canvas.create_text(x+1, y+1, text=glyph, font=('Segoe UI Symbol', int(s*0.72)), fill='black', tags='piece')
						self.canvas.create_text(x, y, text=glyph, font=('Segoe UI Symbol', int(s*0.72)), fill='white', tags='piece')
					else:
						# Black piece: draw subtle white shadow then black glyph (full black requested)
						self.canvas.create_text(x+1, y+1, text=glyph, font=('Segoe UI Symbol', int(s*0.72)), fill='white', tags='piece')
						self.canvas.create_text(x, y, text=glyph, font=('Segoe UI Symbol', int(s*0.72)), fill='black', tags='piece')

	def update_status(self):
		turn = 'White' if self.board.turn else 'Black'
		move_no = self.board.fullmove_number
		self.status_var.set(f"Turn: {turn} | Move: {move_no}")

	def show_overlay(self, text: str, fg: str = 'white', bg: str = '#222222', persist: bool = False):
		"""Draw a centered overlay with `text` over the board area.

		If `persist` is False the overlay is removed after 1400 ms.
		"""
		# remove previous overlay if any
		self.canvas.delete('overlay')
		margin = 20
		s = self.square_size
		x1 = margin
		y1 = margin
		x2 = margin + 8 * s
		y2 = margin + 8 * s

		# dim the board by drawing a stippled rectangle over it
		# stipple gives the impression of transparency on many Tk themes
		self.canvas.create_rectangle(x1, y1, x2, y2, fill='#000000', stipple='gray25', outline='', tags='overlay')
		cx = (x1 + x2) / 2
		cy = (y1 + y2) / 2
		# a centered panel behind the text
		pad = 24
		panel_x1 = cx - self.square_size * 1.5 - pad
		panel_x2 = cx + self.square_size * 1.5 + pad
		panel_y1 = cy - self.square_size * 0.8 - pad
		panel_y2 = cy + self.square_size * 0.8 + pad
		self.canvas.create_rectangle(panel_x1, panel_y1, panel_x2, panel_y2, fill=bg, outline='', tags='overlay')
		# decorative crown for win (only visible if persist)
		if persist:
			crown_y = panel_y1 - 12
			self.canvas.create_text(cx, crown_y, text='\u2654', font=('Helvetica', int(self.square_size*0.6)), fill=fg, tags='overlay')

		# shadow for text (simple offset)
		# choose font size: larger for persistent (win), smaller for transient (check)
		if persist:
			font_size = max(20, int(self.square_size * 1.2))
		else:
			font_size = max(14, int(self.square_size * 0.65))
		shadow_offset = 3
		self.canvas.create_text(cx + shadow_offset, cy + shadow_offset, text=text, font=('Helvetica', font_size, 'bold'), fill='black', tags='overlay')
		self.canvas.create_text(cx, cy, text=text, font=('Helvetica', font_size, 'bold'), fill=fg, tags='overlay')

		if not persist:
			# remove overlay after timeout
			self.root.after(1400, lambda: self.canvas.delete('overlay'))
		else:
			# for persistent win overlay also run fireworks visual + sound
			self._start_fireworks()
			# auto-hide persistent win overlay after ~3 seconds
			# (still runs fireworks and sound immediately)
			self.root.after(4000, lambda: self.clear_overlay())

	def clear_overlay(self):
		self.canvas.delete('overlay')
		self.canvas.delete('fire')

	def play_sound(self, kind: str):
		"""Play a simple sound if available. kind in {'check','win'}."""
		if winsound is None:
			return
		try:
			if kind == 'check':
				# simple system sound
				winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
			elif kind == 'win':
				# play a short clap + fireworks-like beep sequence in background
				threading.Thread(target=self._clap_and_fireworks, daemon=True).start()
			else:
				winsound.MessageBeep()
		except Exception:
			pass

	def _fireworks_beep(self):
		if winsound is None:
			return
		# a simple sequence of beeps with varying frequency to simulate fireworks
		seq = [ (800,120), (1000,100), (600,80), (1200,140), (900,100) ]
		for freq,dur in seq:
			try:
				winsound.Beep(freq, dur)
			except Exception:
				pass
			# short gap between beeps to make them distinct
			time.sleep(0.03)
		# a final short flourish
		try:
			winsound.Beep(1400, 140)
		except Exception:
			pass

	def _clap_and_fireworks(self):
		"""Play a short sequence that sounds like hand-claps, then fireworks beeps."""
		if winsound is None:
			return
		# quick double beeps to simulate claps
		clap_seq = [ (700,60), (700,60), (650,60), (650,60) ]
		for freq,dur in clap_seq:
			try:
				winsound.Beep(freq, dur)
			except Exception:
				pass
			time.sleep(0.06)
		# then run the fireworks beeps
		self._fireworks_beep()

	def _start_fireworks(self):
		# visual fireworks: spawn particles and animate them
		self.canvas.delete('fire')
		cx = (20 + (20 + 8*self.square_size)) / 2
		cy = (20 + (20 + 8*self.square_size)) / 2
		colors = ['#FF5252', '#FFEB3B', '#FF4081', '#448AFF', '#69F0AE']
		particles = []

		# spawn more particles for a longer celebration
		for i in range(30):
			r = random.uniform(4, 14)
			ag = random.choice(colors)
			a = random.uniform(0, 2*3.14159)
			vx = r * math.cos(a)
			vy = r * math.sin(a)
			item = self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill=ag, outline='', tags='fire')
			particles.append((item, vx, vy))

		# run longer animation
		steps = 40

		def step(i):
			for idx, (item, vx, vy) in enumerate(particles):
				self.canvas.move(item, vx, vy)
				# slowly shrink
				coords = self.canvas.coords(item)
				if len(coords) == 4:
					x1,y1,x2,y2 = coords
					cxp = (x1+x2)/2
					sz = max(0.5, (x2-x1) * 0.9)
					self.canvas.coords(item, cxp-sz/2, (y1+y2)/2 - sz/2, cxp+sz/2, (y1+y2)/2 + sz/2)
			if i < steps:
				self.root.after(80, lambda: step(i+1))
			else:
				self.canvas.delete('fire')

		# play sound and run visual
		self.play_sound('win')
		step(0)

	def start_game(self):
		if not self.is_running:
			self.is_running = True
			self.start_btn.config(text='Pause', bg='#FFB74D')
			self.run_loop()
		else:
			self.is_running = False
			self.start_btn.config(text='Start', bg='#4CAF50')

	def reset(self):
		self.is_running = False
		self.start_btn.config(text='Start', bg='#4CAF50')
		self.board = chess.Board()
		self.draw_board()
		self.draw_pieces()
		self.clear_overlay()
		self.update_status()

	def step(self):
		if self.board.is_game_over():
			self.update_status()
			return
		agent = self.white_agent if self.board.turn else self.black_agent
		move = agent.get_action(self.board)
		if move is None:
			self.is_running = False
			return
		from_sq = move.from_square
		to_sq = move.to_square
		piece = self.board.piece_at(from_sq)
		glyph = UNICODE_PIECES.get(piece.symbol(), '?') if piece else '?'
		# push move to update internal board state
		try:
			self.board.push(move)
		except Exception as e:
			print('Invalid move from agent:', e)
			self.is_running = False
			return

		# simple instantaneous update (no animation)
		self.draw_pieces()
		self.update_status()

		# If the move delivers checkmate, display a final overlay and stop.
		if self.board.is_checkmate():
			# side to move is checkmated -> winner is the opposite side
			winner = 'Black wins' if self.board.turn else 'White wins'
			self.show_overlay(winner, fg='white', bg='#1a1a1a', persist=True)
			self.is_running = False
			self.start_btn.config(text='Start')
			return

		# If the move gives check (not mate), show 'Check!'
		if self.board.is_check():
			checker = 'Black' if not self.board.turn else 'White'
			# show brief check overlay with smaller font and play sound
			self.show_overlay(f'Check! ({checker})', fg='yellow', bg='#333333', persist=False)
			self.play_sound('check')

	def run_loop(self):
		if not self.is_running:
			return
		if self.board.is_game_over():
			outcome = self.board.outcome()
			if outcome is None:
				res = 'Draw'
			elif outcome.winner is True:
				res = 'White wins'
			elif outcome.winner is False:
				res = 'Black wins'
			else:
				res = 'Game over'
			self.status_var.set(f"Game over: {res}")
			self.is_running = False
			self.start_btn.config(text='Start')
			return

		self.step()
		# schedule next move
		self.root.after(self.delay, self.run_loop)


def main():
	root = tk.Tk()
	root.title('Chess Game')
	
	ui = ChessUI(root, white_agent=RandomAgent(), black_agent=MinimaxAgent(), square_size=64, delay=400)

	# style buttons
	ui.start_btn.config(bg='#4CAF50', fg='white', font=('Helvetica', 10, 'bold'), activebackground='#FFB74D')
	ui.step_btn.config(bg='#2196F3', fg='white', font=('Helvetica', 10, 'bold'), activebackground='#64B5F6')
	ui.reset_btn.config(bg='#F44336', fg='white', font=('Helvetica', 10, 'bold'), activebackground='#E57373')
	ui.update_status()
	root.mainloop()


if __name__ == '__main__':
	main()

