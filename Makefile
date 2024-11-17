# initial chroma theme, including light and dark
chroma-css:
	hugo gen chromastyles --style=xcode-dark > assets/css/lib/chroma-dark.css
	hugo gen chromastyles --style=perldoc > assets/css/lib/chroma-light.css
