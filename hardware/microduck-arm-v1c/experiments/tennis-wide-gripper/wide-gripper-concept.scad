echo("CONCEPT ONLY: no shaft/horn holes, retention, tolerance or strength release");
echo("Units mm. Neutral/open pose. Pads are contact references, not printed PA12.");
color("orange") translate([10.0, 26.0, -5.5]) cube([12.0, 31.0, 3.0], center=true);
color("orange") translate([32.5, 44.0, -5.5]) cube([33.0, 8.0, 3.0], center=true);
color("green") translate([48.0, 40.0, 0]) cube([24, 5, 8], center=true);
color("orange") translate([10.0, -26.0, -5.5]) cube([12.0, 31.0, 3.0], center=true);
color("orange") translate([32.5, -44.0, -5.5]) cube([33.0, 8.0, 3.0], center=true);
color("green") translate([48.0, -40.0, 0]) cube([24, 5, 8], center=true);
