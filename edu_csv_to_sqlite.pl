use strict;
use warnings;
use Fcntl;

my $stu_id = 0;
my $val1 = "";
my $val2 = "";
my $val3 = "";
my $val4 = "";
my $val5 = "";
my $val6 = "";
my $val7 = "";
my $val8 = "";

my $parsed = 0;
my $unparsed = 0;
open(FILE, '<:encoding(UTF-8)', 'indicator.csv');
while(<FILE>) {
	if($_ =~ /^(\d+),(\d+|NA),(\d+|NA),(\d+|NA),(\d+|NA),(\d+|NA),(\d+|NA),(\d+|NA),(\d+|NA).*$/) {
		$stu_id = $1;
		$val1 = $2;
		$val2 = $3;
		$val3 = $4;
		$val4 = $5;
		$val5 = $6;
		$val6 = $7;
		$val7 = $8;
		$val8 = $9;
		if($val1 eq "NA") { $val1 = "0"; }
		if($val2 eq "NA") { $val2 = "0"; }
		if($val3 eq "NA") { $val3 = "0"; }
		if($val4 eq "NA") { $val4 = "0"; }
		if($val5 eq "NA") { $val5 = "0"; }
		if($val6 eq "NA") { $val6 = "0"; }
		if($val7 eq "NA") { $val7 = "0"; }
		if($val8 eq "NA") { $val8 = "0"; }
		print "INSERT INTO student(STU_ID, BYS87A, BYS87C, BYS87F, F1S18A, F1S18B, F1S18C, F1S18D, F1S18E) VALUES(".$stu_id.", ".$val1.", ".$val2.", ".$val3.", ".$val4.", ".$val5.", ".$val6.", ".$val7.", ".$val8.");\n";
		$parsed++;
	} else {
		# print "UNPARSED: ".$_;
		$unparsed++;
	}
}
close FILE;

open(FILE, '<:encoding(UTF-8)', 'outcome.csv');
seek FILE, 0, 0;
while(<FILE>) {
	if($_ =~ /^(\d+),(\d+|NA),(\d+|NA),(.*?),(\d+|NA).*$/) {
		$stu_id = $1;
		$val1 = $2;
		$val2 = $3;
		$val3 = $4;
		$val4 = $5;
		if($val1 eq "NA") { $val1 = "-1"; }
		if($val2 eq "NA") { $val2 = "-1"; }
		if($val3 eq "NA") { $val3 = "-1"; }
		if($val4 eq "NA") { $val4 = "-1"; }
		print "UPDATE student SET F2PS1AID=".$val1.", F3TZSTEM1TOT=".$val2.", F3TZSTEM2GPA=".$val3.", CREDGRAD=".$val4." WHERE STU_ID=".$stu_id.";\n";
		print "INSERT OR IGNORE INTO student(STU_ID, F2PS1AID, F3TZSTEM1TOT, F3TZSTEM2GPA, CREDGRAD) VALUES(".$stu_id.", ".$val1.", ".$val2.", ".$val3.", ".$val4.");\n";
		$parsed++;
	} else {
		# print "UNPARSED: ".$_;
		$unparsed++;
	}
}
close FILE;

print "parsed: ".$parsed."\n";
print "unparsed: ".$unparsed."\n";










